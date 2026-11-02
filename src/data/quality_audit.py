"""
Quality audit of question -> reply pairs (read-only). Answers: how much of the data is garbage?

Looks at every customer turn that got a business reply in redacted_conversations.jsonl and flags:
- GREETING_Q   : the customer only said hi/hello/hii/namaste…
- CANNED       : the business reply is a fixed auto-reply (same text repeated many times by that business,
                 or across many businesses)
- DEAD_END     : "number closed / not available / call on this number / we will get back" style replies
- TOO_SHORT    : reply too short to teach anything ("ok", "yes", "haa")
- TOO_LONG     : reply is a wall of text (brochure dumps)
- LOW_INFO_Q   : customer message has < 2 words of real content

Writes logs/<export>_quality_audit.json and data/interim/<export>/quality_examples.txt (examples per flag).

Run from mbg-ai-model/:   python src/data/quality_audit.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_report  # noqa: E402

GREETING = re.compile(
    r"^\W*(hi+|hii+|hello+|hel+o|hlo+|hey+|hai|namaste|namaskar|good\s*(morning|afternoon|evening|night)|"
    r"gm|hy|hlw|helo|ram\s*ram|jai\s*shree\s*ram|radhe\s*radhe|salam|assalam\w*)\W*(sir|mam|madam|ji|bhai|there)?\W*$",
    re.I,
)
DEAD_END = re.compile(
    r"(number (is )?(closed|not in use|band)|not available|currently unavailable|out of office|"
    r"call (us )?on|contact (us )?on|call me on|whatsapp (us )?on|we will get back|will get back to you|"
    r"will contact you|our team will (call|contact)|reach us at|for (more )?(details|info).{0,20}call|"
    r"band hai|baad me|abhi busy|out of station|this number is|is no longer|thank(s| you) for (contacting|reaching|your message))",
    re.I,
)


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\d", "#", t)
    t = re.sub(r"[^\w#ऀ-ॿ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def words(t: str) -> int:
    return len(re.findall(r"[A-Za-zऀ-ॿ]{2,}", t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--canned-per-business", type=int, default=5, help="same reply this many times in one business = canned")
    ap.add_argument("--canned-global", type=int, default=15, help="same reply across this many businesses = canned")
    args = ap.parse_args()
    name = args.name or latest_export("interim")
    d = export_dir("interim", name)
    src = d / "redacted_conversations.jsonl"

    # Pass 1: how often each reply text repeats (per business and across businesses)
    per_biz = collections.Counter()
    biz_sets: dict[str, set] = collections.defaultdict(set)
    for conv in read_jsonl(src):
        b = conv["business_id"]; turns = conv["turns"]
        for i in range(len(turns) - 1):
            if turns[i]["role"] == "customer" and turns[i + 1]["role"] == "business":
                k = norm(turns[i + 1]["text"])[:300]
                per_biz[(b, k)] += 1
                biz_sets[k].add(b)

    # Pass 2: flag every pair
    flags_by_author: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    total_by_author = collections.Counter()
    clean_by_author = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    top_canned = collections.Counter()
    clean_lang_biz = collections.Counter()
    for conv in read_jsonl(src):
        b = conv["business_id"]; turns = conv["turns"]
        for i in range(len(turns) - 1):
            q, a = turns[i], turns[i + 1]
            if not (q["role"] == "customer" and a["role"] == "business"):
                continue
            author = "ai" if "ai" in a["authors"] else a["authors"][0]
            total_by_author[author] += 1
            k = norm(a["text"])[:300]
            f = []
            if GREETING.match(q["text"].strip()):
                f.append("GREETING_Q")
            if words(q["text"]) < 2 and "GREETING_Q" not in f:
                f.append("LOW_INFO_Q")
            if per_biz[(b, k)] >= args.canned_per_business or len(biz_sets[k]) >= args.canned_global:
                f.append("CANNED"); top_canned[k[:120]] += 1
            if DEAD_END.search(a["text"]):
                f.append("DEAD_END")
            if len(a["text"]) < 15 or words(a["text"]) < 3:
                f.append("TOO_SHORT")
            if len(a["text"]) > 1200:
                f.append("TOO_LONG")
            for x in f:
                flags_by_author[author][x] += 1
                if len(examples[x]) < 15 and hash((b, k)) % 97 == 0:
                    examples[x].append(f"[{author}] Q: {q['text'][:160]!r}\n        A: {a['text'][:220]!r}")
            if not f:
                clean_by_author[author] += 1
                clean_lang_biz[b] += 1
                if len(examples["CLEAN"]) < 25 and hash((b, k)) % 53 == 0:
                    examples["CLEAN"].append(f"[{author}] Q: {q['text'][:160]!r}\n        A: {a['text'][:260]!r}")

    total = sum(total_by_author.values()); clean = sum(clean_by_author.values())
    report = {
        "export": name,
        "pairs_total": total, "pairs_clean": clean, "clean_pct": round(100 * clean / max(1, total), 1),
        "pairs_by_author": dict(total_by_author.most_common()),
        "clean_by_author": dict(clean_by_author.most_common()),
        "flags_by_author": {a: dict(c.most_common()) for a, c in flags_by_author.items()},
        "businesses_with_clean_pairs": len(clean_lang_biz),
        "top_canned_replies": top_canned.most_common(25),
    }
    write_report(f"{name}_quality_audit", report)
    with (d / "quality_examples.txt").open("w", encoding="utf-8") as fh:
        for k, v in examples.items():
            fh.write(f"\n===== {k} =====\n" + "\n".join(v) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "top_canned_replies"}, indent=2, ensure_ascii=False))
    print("\nTOP CANNED REPLIES (times used):")
    for t, n in report["top_canned_replies"][:15]:
        print(f"  {n:>6}  {t}")


if __name__ == "__main__":
    main()
