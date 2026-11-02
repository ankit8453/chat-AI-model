"""
Level 1 · Step 2 — Clean and filter conversations.

Input : data/interim/<export>/conversations.jsonl         (from step 01)
Output: data/interim/<export>/clean_conversations.jsonl
        logs/<export>_02_clean.json

Message level : drop platform/system noise ("AI reply is temporarily unavailable…"), empty text,
                emoji/number/punctuation-only messages.
Turn level    : merge back-to-back messages from the same side into one turn.
Chat level    : keep chats where at least one customer turn is answered by a business turn;
                drop very long chats (> --max-messages, usually internal staff/group chats).

Run from mbg-ai-model/:
    python src/data/02_clean_filter.py                 # latest export
    python src/data/02_clean_filter.py --name messages_2026-10-31
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_report  # noqa: E402
from common.text import has_letters, is_system_noise, language_hint  # noqa: E402


def clean_messages(msgs: list[dict], c: collections.Counter) -> list[dict]:
    kept = []
    for m in msgs:
        t = m["text"]
        if is_system_noise(t):
            c["msg_drop_system_noise"] += 1; continue
        if not has_letters(t):
            c["msg_drop_no_letters"] += 1; continue
        kept.append(m)
    return kept


def merge_turns(msgs: list[dict]) -> list[dict]:
    """Consecutive messages from the same side -> one turn. Keeps every author that contributed, and
    the individual messages as `parts` (author + text) so a reply can later be cut at the first message
    from a different author (fixes AI replies getting glued to later staff messages)."""
    turns: list[dict] = []
    for m in msgs:
        part = {"author": m["author"], "text": m["text"]}
        if turns and turns[-1]["role"] == m["role"]:
            t = turns[-1]
            t["text"] += "\n" + m["text"]
            t["parts"].append(part)
            if m["author"] not in t["authors"]:
                t["authors"].append(m["author"])
            t["ts_end"] = m["ts"]
        else:
            turns.append({"role": m["role"], "authors": [m["author"]], "ts": m["ts"],
                          "ts_end": m["ts"], "text": m["text"], "parts": [part]})
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    ap.add_argument("--max-messages", type=int, default=400,
                    help="Drop chats with more raw messages than this (internal/group chats)")
    args = ap.parse_args()

    name = args.name or latest_export("interim")
    d = export_dir("interim", name)
    src = d / "conversations.jsonl"
    out = d / "clean_conversations.jsonl"

    c = collections.Counter()
    answered_by = collections.Counter()
    cust_lang = collections.Counter()
    biz_with_pairs = set()
    with out.open("w", encoding="utf-8") as f:
        for conv in read_jsonl(src):
            c["chats_in"] += 1
            if len(conv["messages"]) > args.max_messages:
                c["chat_drop_too_long"] += 1; continue
            msgs = clean_messages(conv["messages"], c)
            turns = merge_turns(msgs)
            # customer turn immediately followed by a business turn = an answered question
            pairs = 0
            for i in range(len(turns) - 1):
                if turns[i]["role"] == "customer" and turns[i + 1]["role"] == "business":
                    pairs += 1
                    for a in turns[i + 1]["authors"]:
                        answered_by[a] += 1
            if pairs == 0:
                c["chat_drop_no_answered_question"] += 1; continue
            for t in turns:
                if t["role"] == "customer":
                    cust_lang[language_hint(t["text"])] += 1
            biz_with_pairs.add(conv["uid"])
            biz_authors = {a for t in turns if t["role"] == "business" for a in t["authors"]}
            conv_out = {
                "export": conv["export"], "chat_id": conv["chat_id"], "uid": conv["uid"],
                "n_turns": len(turns), "n_answered": pairs,
                "has_ai": "ai" in biz_authors, "business_authors": sorted(biz_authors),
                "turns": turns,
            }
            f.write(json.dumps(conv_out, ensure_ascii=False) + "\n")
            c["chats_kept"] += 1
            c["answered_pairs"] += pairs
            if conv_out["has_ai"]:
                c["chats_with_ai"] += 1

    report = {"export": name, **c, "businesses_with_answered_chats": len(biz_with_pairs),
              "answered_pairs_by_business_author": dict(answered_by.most_common()),
              "customer_turn_language": dict(cust_lang.most_common())}
    write_report(f"{name}_02_clean", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
