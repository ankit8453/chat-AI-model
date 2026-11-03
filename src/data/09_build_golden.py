"""
Level 2 · Golden test set — the fixed exam every model version is scored on.

Input : data/processed/<export>/test.jsonl      (step 8; businesses never seen in training)
        data/interim/<export>/sft_items.jsonl   (for each row's memory text + hand-over message)
Output: data/processed/<export>/golden.jsonl    -> private HF dataset ankit-iiitdmj/mbg-sft-<export>/golden.jsonl

300 rows are picked (about 30% hand-overs, every language, Hindi oversampled, at most 8 per business) and
reviewed by Claude; the rows whose reference is wrong are listed in EXCLUDE and dropped after picking, so every
golden row is a reviewed one (the exam scores the hand-over decision against the reference, so it must be right).
Excluded: answers where a hand-over was right (payments, "any update?" on an existing order, staff/personal
chats, delivery scheduling), Marathi / Gujarati / Punjabi messages (no answers trained for them), references
with a placeholder, the customer's name or an invented promise.

Each row: {item_id, business_id, lang, topic, kind, messages (prompt only), reference, handover, source}

Run from mbg-ai-model/:   python src/data/09_build_golden.py [--no-push]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_jsonl  # noqa: E402

TARGET = 300
HANDOVER_SHARE = 0.30
HINDI_MIN = 30
PER_BUSINESS = 8
SEED = 7
EXCLUDE = {   # reviewed by Claude 2026-11-03 (messages_2026-10-31): 43 of 300
    "0d51f438201b33a7", "148584e408f53df3", "1d6ea664cf1a4f29", "3715c67946d3b22a", "41ccef38c8fbad12",
    "428fdff34ee1e8e6", "48ad2a43b92d302c", "4cccb26fed624f58", "51ab5773389be561", "5330942db357de12",
    "5cf54f90af334f4c", "60f6cd9576035947", "6bb56cc36a5bb603", "6e261af0ba8f68fb", "7a70bd7af4df9d33",
    "7c34b3071f2b7228", "7e7e8afedf5b1c0e", "7fb35fa30616bc3a", "853ffb59e93d3390", "86fa0bde85db77eb",
    "8880bf8e6a822382", "8fad1a8985fc2f5f", "901c46baee6840e3", "91f5208a4275a602", "9712e6c69d3398cd",
    "971d2cf31a5ee333", "972eb699588e1766", "9ce492fcd45512ef", "9e614594a715a7fa", "a50ac7f9d2b6ad18",
    "a9ebba3f54a41fcb", "aa6631cb4895f9bb", "ac2e0ca8c303e6a5", "ca832ca0b9163482", "d017571761c96406",
    "d9972b55f019cdba", "db89f014a677bd43", "e2090c2b95b055be", "e9a06d02d873d5c1", "eeef931e65b3ad9e",
    "f14c6ff768c745a8", "faaf1a0c80f96eb6", "fc0d5edbcc67844e",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    name = args.name or latest_export("interim")

    items = {it["item_id"]: it for it in read_jsonl(export_dir("interim", name) / "sft_items.jsonl")}
    out = export_dir("processed", name)
    pool = []
    for r in read_jsonl(out / "test.jsonl"):
        it = items[r["item_id"]]
        pool.append({"item_id": r["item_id"], "business_id": r["business_id"], "lang": r["lang"],
                     "topic": r["topic"], "kind": r["kind"], "messages": r["messages"][:-1],
                     "reference": r["messages"][-1]["content"], "handover": it["handover"],
                     "source": it["source"]})
    rng = random.Random(SEED)
    rng.shuffle(pool)

    per_biz: collections.Counter = collections.Counter()
    picked: list[dict] = []

    def take(rows: list[dict], n: int) -> None:
        for r in rows:
            if n <= 0:
                return
            if r in picked or per_biz[r["business_id"]] >= PER_BUSINESS:
                continue
            picked.append(r)
            per_biz[r["business_id"]] += 1
            n -= 1

    take([r for r in pool if r["lang"] == "hindi"], HINDI_MIN)
    n_ho = round(TARGET * HANDOVER_SHARE) - sum(r["kind"] == "handover" for r in picked)
    take([r for r in pool if r["kind"] == "handover"], n_ho)
    take([r for r in pool if r["kind"] == "answer"], TARGET - len(picked))
    picked = [r for r in picked if r["item_id"] not in EXCLUDE]   # after picking: only reviewed rows remain

    write_jsonl(out / "golden.jsonl", picked)
    print(json.dumps({
        "rows": len(picked), "businesses": len(per_biz),
        "kind": dict(collections.Counter(r["kind"] for r in picked)),
        "lang": dict(collections.Counter(r["lang"] for r in picked)),
        "topic": dict(collections.Counter(r["topic"] for r in picked).most_common()),
    }, indent=2))

    if args.no_push:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/mbg-sft-{name}"
    api.upload_file(path_or_fileobj=str(out / "golden.jsonl"), path_in_repo="golden.jsonl", repo_id=repo_id,
                    repo_type="dataset", commit_message="golden test set")
    print(f"pushed -> {repo_id}/golden.jsonl  private={api.repo_info(repo_id, repo_type='dataset').private}")


if __name__ == "__main__":
    main()
