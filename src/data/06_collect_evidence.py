"""
Level 1 · Step 6a — Collect each business's own facts ("evidence") for memory building.

Input : data/interim/<export>/redacted_conversations.jsonl   (from step 03)
        data/interim/<export>/judged_all.jsonl                (Gate 3 result, downloaded from HF if missing)
Output: data/interim/<export>/business_evidence.jsonl          (one row per business with kept questions)
        logs/<export>_06_evidence.json

For every business that has at least one real customer question (Gate 3 keep_question), we pick the
business messages that carry facts — prices, timings, addresses, links, offers, menus — together with the
customer message they answered, so the teacher model can see what each fact refers to. Duplicates are
removed and recent messages win ties (prices and offers change). The teacher model (notebook
02_build_memories) turns this evidence into one memory in our format (config/memory_schema.json).

Run from mbg-ai-model/:   python src/data/06_collect_evidence.py
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_jsonl, write_report  # noqa: E402
from common.quality import norm  # noqa: E402

MAX_CHARS = 12_000        # evidence budget per business (≈ 3–5k tokens; the notebook trims to fit in tokens)
Q_CHARS, A_CHARS = 200, 700
MIN_SCORE = 2

# Signals that a business message states a fact worth remembering.
FACT_PATTERNS = [
    (re.compile(r"(₹|\brs\.?\s?\d|\binr\b|\d+\s?/-|\bprice\b|\brate\b|\bfees?\b|\bcharges?\b)", re.I), 3),
    (re.compile(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b|\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b|\btimings?\b", re.I), 2),
    (re.compile(r"\b(road|nagar|near|opp\.?|opposite|floor|shop no|plot|sector|market|chowk|complex)\b|\b\d{6}\b", re.I), 2),
    (re.compile(r"https?://|www\.", re.I), 1),
    (re.compile(r"\b(offer|discount|% ?off|free|emi|warranty|delivery|refund|booking|appointment)\b", re.I), 1),
    (re.compile(r"\b(we (offer|provide|have|deal)|our (services|products|menu|courses?|packages?))\b|^\s*[•\-*]\s", re.I | re.M), 2),
]


def fact_score(text: str) -> int:
    return sum(w for p, w in FACT_PATTERNS if p.search(text))


def business_parts(turn: dict) -> list[str]:
    parts = turn.get("parts") or [{"author": turn["authors"][0], "text": turn["text"]}]
    return [p["text"] for p in parts if p.get("text")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    args = ap.parse_args()
    name = args.name or latest_export("interim")
    d = export_dir("interim", name)

    judged_path = d / "judged_all.jsonl"
    if not judged_path.exists():
        import shutil
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(f"ankit-iiitdmj/mbg-judge-{name}", "judged_all.jsonl", repo_type="dataset"),
                    judged_path)

    kept_q = collections.Counter()
    for r in read_jsonl(judged_path):
        if r["keep_question"]:
            kept_q[r["business_id"]] += 1
    print(f"businesses with kept questions: {len(kept_q):,}")

    # business_id -> {norm_key: (score, ts, question, reply)}
    pool: dict[str, dict[str, tuple]] = collections.defaultdict(dict)
    for conv in read_jsonl(d / "redacted_conversations.jsonl"):
        b = conv["business_id"]
        if b not in kept_q:
            continue
        turns = conv["turns"]
        for i, t in enumerate(turns):
            if t["role"] != "business":
                continue
            q = turns[i - 1]["text"] if i > 0 and turns[i - 1]["role"] == "customer" else ""
            for text in business_parts(t):
                s = fact_score(text)
                if s < MIN_SCORE:
                    continue
                k = norm(text)[:200]
                ts = t.get("ts") or ""
                old = pool[b].get(k)
                if old is None or ts > old[1]:
                    pool[b][k] = (s, ts, q, text)

    rows, sizes = [], []
    for b, n_q in kept_q.most_common():
        items = sorted(pool.get(b, {}).values(), key=lambda x: (x[0], x[1]), reverse=True)
        snippets, used = [], 0
        for s, ts, q, a in items:
            snip = {"date": ts[:7], "customer": q[:Q_CHARS], "business": a[:A_CHARS]}
            size = len(snip["customer"]) + len(snip["business"]) + 30
            if used + size > MAX_CHARS:
                continue
            snippets.append(snip)
            used += size
        if not snippets:
            continue
        rows.append({"business_id": b, "kept_questions": n_q, "candidate_snippets": len(items),
                     "snippets": snippets, "chars": used})
        sizes.append(used)

    out = d / "business_evidence.jsonl"
    write_jsonl(out, rows)
    sizes.sort()
    report = {
        "export": name, "businesses_with_kept_questions": len(kept_q), "businesses_with_evidence": len(rows),
        "no_evidence": len(kept_q) - len(rows),
        "snippets_total": sum(len(r["snippets"]) for r in rows),
        "chars_p50": sizes[len(sizes) // 2] if sizes else 0, "chars_max": sizes[-1] if sizes else 0,
        "businesses_under_1000_chars": sum(1 for s in sizes if s < 1000),
    }
    p = write_report(f"{name}_06_evidence", report)
    print(report)
    print(f"wrote {out} ({len(rows):,} businesses) · report {p}")


if __name__ == "__main__":
    main()
