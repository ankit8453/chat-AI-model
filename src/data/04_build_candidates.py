"""
Level 1 · Step 4 — Build judge candidates (Gate 1 rules + Gate 2 clean reply cut).

Input : data/interim/<export>/redacted_conversations.jsonl   (from step 03)
Output: data/interim/<export>/candidates.jsonl                 (pairs to send to the AI judge, Gate 3)
        logs/<export>_04_candidates.json

Gate 2 — clean reply: the reply is ONLY the business messages sent right after the customer's question
         by the same author as the first reply message (stops at the first message from someone else).
         This fixes AI replies that were glued to later staff messages.
Gate 1 — rules: drop greeting-only / low-info questions, canned auto-replies, dead-end replies
         ("number closed / call on …"), too short / too long replies.
Selection (what goes to the judge):
         - AI / bot / agent replies that pass Gate 1
         - staff (unknown) replies that pass Gate 1 AND whose question looks like an enquiry
           (most other staff pairs are internal / personal chats)
         - flow replies are excluded (scripted menus, not answers)

Each candidate carries the 2 turns before the question as context for the judge.

Run from mbg-ai-model/:   python src/data/04_build_candidates.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_report  # noqa: E402
from common.quality import ENQUIRY_HINT, gate1_flags, norm  # noqa: E402
from common.text import language_hint  # noqa: E402

AUTHOR_LABEL = {"ai": "ai", "bot": "bot", "agent": "agent", "unknown": "staff", "flow": "flow"}


def clean_reply(turn: dict) -> tuple[str, str]:
    """Gate 2: leading parts by the same author as the first part. Returns (author, text)."""
    parts = turn.get("parts") or [{"author": turn["authors"][0], "text": turn["text"]}]
    first = parts[0]["author"]
    kept = []
    for p in parts:
        if p["author"] != first:
            break
        kept.append(p["text"])
    return first, "\n".join(kept)


def iter_pairs(conv: dict):
    turns = conv["turns"]
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "customer" and turns[i + 1]["role"] == "business":
            author, reply = clean_reply(turns[i + 1])
            yield i, turns[i]["text"], author, reply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    ap.add_argument("--context-turns", type=int, default=2)
    args = ap.parse_args()

    name = args.name or latest_export("interim")
    d = export_dir("interim", name)
    src = d / "redacted_conversations.jsonl"

    # Pass 1: reply repetition (per business, and how many businesses share a reply)
    per_biz = collections.Counter()
    biz_sets: dict[str, set] = collections.defaultdict(set)
    for conv in read_jsonl(src):
        for _, _, _, reply in iter_pairs(conv):
            k = norm(reply)[:300]
            per_biz[(conv["business_id"], k)] += 1
            biz_sets[k].add(conv["business_id"])

    # Pass 2: gates + selection
    c = collections.Counter()
    flags_all = collections.Counter()
    by_author = collections.Counter()
    biz_kept = collections.Counter()
    langs = collections.Counter()
    out = d / "candidates.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for conv in read_jsonl(src):
            b = conv["business_id"]
            turns = conv["turns"]
            for i, question, author_raw, reply in iter_pairs(conv):
                c["pairs_total"] += 1
                author = AUTHOR_LABEL.get(author_raw, author_raw)
                if author == "flow":
                    c["drop_flow"] += 1; continue
                k = norm(reply)[:300]
                flags = gate1_flags(question, reply,
                                    reply_repeats_in_business=per_biz[(b, k)],
                                    reply_business_count=len(biz_sets[k]))
                if flags:
                    for x in flags:
                        flags_all[x] += 1
                    c["drop_gate1"] += 1; continue
                if author == "staff" and not ENQUIRY_HINT.search(question):
                    c["drop_staff_not_enquiry"] += 1; continue
                ctx = [{"role": t["role"], "text": t["text"][:500]}
                       for t in turns[max(0, i - args.context_turns):i]]
                pid = hashlib.sha256(f"{conv['conversation_id']}|{i}".encode()).hexdigest()[:16]
                lang = language_hint(question)
                f.write(json.dumps({
                    "pair_id": pid, "export": conv["export"], "business_id": b,
                    "conversation_id": conv["conversation_id"], "turn_index": i,
                    "author": author, "lang_hint": lang,
                    "context": ctx, "question": question[:1500], "reply": reply[:1500],
                }, ensure_ascii=False) + "\n")
                c["candidates"] += 1
                by_author[author] += 1
                biz_kept[b] += 1
                langs[lang] += 1

    report = {"export": name, **c, "gate1_flags": dict(flags_all.most_common()),
              "candidates_by_author": dict(by_author.most_common()),
              "candidates_by_language_hint": dict(langs.most_common()),
              "businesses": len(biz_kept),
              "max_candidates_one_business": max(biz_kept.values()) if biz_kept else 0}
    write_report(f"{name}_04_candidates", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
