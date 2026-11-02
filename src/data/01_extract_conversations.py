"""
Level 1 · Step 1 — Extract text conversations from a Chatrace export.

Input : a PostgreSQL custom-format dump containing public.messages (e.g. messages_2026-10-31.dump)
Output: data/interim/<export>/conversations.jsonl   (one line per chat, messages in time order)
        logs/<export>_01_extract.json                (counts)

Streams the dump through `pg_restore -a -t messages -f -` (no database restore), keeps text-bearing
messages (text + interactive), labels who wrote each outgoing message, and groups by chat using a
temporary SQLite file so the 3M+ rows never sit in memory.

Run from mbg-ai-model/:
    python src/data/01_extract_conversations.py --dump ../messages_2026-10-31.dump
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, write_report  # noqa: E402
from common.text import normalise  # noqa: E402

COLS = ["id", "uid", "chat_id", "type", "message_id", "status", "body", "route",
        "timestamp", "createdAt", "updatedAt", "error"]
KEEP_TYPES = {"text", "interactive"}


def unescape(v: str):
    """Undo PostgreSQL COPY text-format escaping."""
    if v == r"\N":
        return None
    return (v.replace("\\\\", "\x00").replace("\\t", "\t").replace("\\n", "\n")
             .replace("\\r", "\r").replace("\x00", "\\"))


def extract_text(msg_type: str, route: str, body: dict) -> str | None:
    """Pull the human-readable text out of a message body."""
    if msg_type == "text":
        t = body.get("text")
        if isinstance(t, dict):
            t = t.get("body")
        return t if isinstance(t, str) else None
    # interactive
    if route == "INCOMING":
        for k in ("button_reply", "list_reply"):
            v = body.get(k)
            if isinstance(v, dict) and isinstance(v.get("title"), str):
                return v["title"]
        t = body.get("text")
        return t if isinstance(t, str) else None
    # outgoing interactive: the message body plus the option labels
    b = body.get("body")
    text = b.get("text") if isinstance(b, dict) else (b if isinstance(b, str) else body.get("text"))
    if not isinstance(text, str):
        return None
    opts = []
    action = body.get("action") if isinstance(body.get("action"), dict) else {}
    for btn in action.get("buttons", []) or []:
        title = (btn.get("reply") or {}).get("title") if isinstance(btn, dict) else None
        if title:
            opts.append(title)
    for sec in action.get("sections", []) or []:
        for row in (sec.get("rows", []) if isinstance(sec, dict) else []):
            if isinstance(row, dict) and row.get("title"):
                opts.append(row["title"])
    return text + ("\n[options: " + " | ".join(opts) + "]" if opts else "")


def author_of(route: str, msg_type: str, body: dict) -> str:
    """Who wrote an outgoing message: ai | flow | bot | agent | unknown. Incoming -> customer."""
    if route == "INCOMING":
        return "customer"
    sent_by = body.get("sentBy") if isinstance(body.get("sentBy"), dict) else {}
    if body.get("flow_node_id") or sent_by.get("flow"):
        return "flow"
    if body.get("chatbot"):
        return "ai"
    if body.get("origin") == "bot":
        return "bot"
    if sent_by:
        return "agent"
    return "unknown"      # mostly typed by staff on the phone / WhatsApp app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="Path to the Chatrace .dump file")
    ap.add_argument("--name", help="Export name (default: dump file name without extension)")
    args = ap.parse_args()

    dump = Path(args.dump).resolve()
    if not dump.exists():
        raise SystemExit(f"Dump not found: {dump}")
    name = args.name or dump.stem
    out_dir = export_dir("interim", name)
    db_path = out_dir / "_stage.sqlite"
    if db_path.exists():
        db_path.unlink()

    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
    db.execute("""CREATE TABLE m (id INTEGER, chat_id INTEGER, uid TEXT, ts TEXT,
                  route TEXT, author TEXT, text TEXT)""")

    counts = collections.Counter()
    authors = collections.Counter()
    started = time.time()
    proc = subprocess.Popen(["pg_restore", "-a", "-t", "messages", "-f", "-", str(dump)],
                            stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    in_copy = False
    batch = []
    for line in proc.stdout:
        if not in_copy:
            in_copy = line.startswith("COPY public.messages")
            continue
        if line.startswith("\\."):
            break
        counts["rows_total"] += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) != len(COLS):
            counts["rows_malformed"] += 1
            continue
        r = dict(zip(COLS, (unescape(p) for p in parts)))
        if r["type"] not in KEEP_TYPES or not r["body"]:
            counts["rows_skipped_type"] += 1
            continue
        try:
            body = json.loads(r["body"])
        except json.JSONDecodeError:
            counts["rows_bad_json"] += 1
            continue
        if not isinstance(body, dict):
            counts["rows_bad_json"] += 1
            continue
        text = extract_text(r["type"], r["route"], body)
        if not text or not text.strip():
            counts["rows_empty_text"] += 1
            continue
        author = author_of(r["route"], r["type"], body)
        authors[author] += 1
        batch.append((int(r["id"]), int(r["chat_id"]), r["uid"], r["createdAt"] or "",
                      r["route"], author, normalise(text)))
        if len(batch) >= 50_000:
            db.executemany("INSERT INTO m VALUES (?,?,?,?,?,?,?)", batch); batch.clear()
            print(f"  … {counts['rows_total']:,} rows read", flush=True)
    if batch:
        db.executemany("INSERT INTO m VALUES (?,?,?,?,?,?,?)", batch)
    proc.stdout.close()
    if proc.wait() != 0:
        raise SystemExit("pg_restore failed — is the dump complete and pg_restore installed?")
    db.commit()
    counts["rows_kept"] = sum(authors.values())

    print("Grouping into conversations …", flush=True)
    db.execute("CREATE INDEX ix ON m(chat_id, ts, id)")
    out = out_dir / "conversations.jsonl"
    n_chats = 0
    with out.open("w", encoding="utf-8") as f:
        cur_chat, conv = None, None
        for chat_id, uid, ts, route, author, text in db.execute(
                "SELECT chat_id, uid, ts, route, author, text FROM m ORDER BY chat_id, ts, id"):
            if chat_id != cur_chat:
                if conv:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n"); n_chats += 1
                cur_chat = chat_id
                conv = {"export": name, "chat_id": chat_id, "uid": uid, "messages": []}
            conv["messages"].append({"role": "customer" if route == "INCOMING" else "business",
                                     "author": author, "ts": ts, "text": text})
        if conv:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n"); n_chats += 1
    db.close()
    db_path.unlink()

    report = {"export": name, "dump": str(dump), "seconds": round(time.time() - started, 1),
              **counts, "conversations": n_chats, "messages_by_author": dict(authors.most_common())}
    write_report(f"{name}_01_extract", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
