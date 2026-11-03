"""
Level 1 · Step 8 — Fact-check the teacher replies, split by business, push the fine-tuning dataset.

Input : data/interim/<export>/sft_items.jsonl                 (step 7)
        ankit-iiitdmj/mbg-answers-<export>/answers_all.jsonl   (notebook 03, downloaded if missing)
Output: data/processed/<export>/{train,validation,test}.jsonl + summary.json
        -> private HF dataset ankit-iiitdmj/mbg-sft-<export>   (skip with --no-push)

Every teacher reply must pass: facts grounded in the memory chunks it was given (every link, number of 3+
digits and clock time), no placeholder tags, no customer name, right script for the language, not too long, no word loops. Answers to
Marathi / Bengali customer messages are dropped (the teacher writes those languages badly); hand-overs are kept.
A hand-over is normalised to exactly "[HANDOVER]\\n<hand-over message>". Rows are split by business
(never by row), so validation/test businesses are unseen during training.

Each output row: {"messages": [system, ...history, user, assistant], "item_id", "business_id", "kind", "lang", "topic"}

Run from mbg-ai-model/:   python src/data/08_assemble_split_push.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_jsonl, write_report  # noqa: E402
from sft.build import (HANDOVER_TAG, LEAKS, bold_to_whatsapp, is_handover, other_language,  # noqa: E402
                       reply_problems)

# Complaints / refunds must be handed over (system prompt rule 2); an answer to one is not used as a target.
COMPLAINT = re.compile(
    r"\b(refund|complain\w*|not working|nahi chal|kaam nahi kar|kharab|kharaab|damage\w*|broken|defect\w*|"
    r"wrong (?:item|product|order)|galat|abhi tak|still not|not (?:yet )?(?:received|delivered|come|came)|"
    r"nahi (?:aya|aaya|ayi|aayi|mila|pahuncha)|cancel\w*|paisa wapas|money back|fraud|cheat\w*|worst|bekar|bakwas)\b"
    r"|शिकायत|रिफंड|खराब|नहीं आया|नहीं मिला", re.I)


def split_of(business_id: str) -> str:
    h = int(hashlib.md5(business_id.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "validation" if h < 10 else "train"


def clean_answer(text: str) -> str:
    t = (text or "").strip().strip('"').strip()
    for prefix in ("Reply:", "Answer:", "REPLY:", "Assistant:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return bold_to_whatsapp(t)


def target_for(item: dict, answer: str) -> tuple[str | None, str]:
    """(target or None, reason)."""
    handover = item["handover"]
    a = clean_answer(answer)
    if not a:
        return None, "empty"
    if is_handover(a, handover):
        # The teacher decided to hand over; the reply is always exactly the hand-over message.
        return f"{HANDOVER_TAG}\n{handover}", "ok_handover"
    if re.search(r"hand\s*-?over", a, re.I):
        return None, "handover_tag_mid_text"
    if a.startswith("["):
        return None, "bracket_header"
    if any(w in a.lower() for w in LEAKS):
        return None, "mentions_rules"
    q = item["messages"][-1]["content"]
    if other_language(q):
        return None, "marathi_bengali_answer"
    if COMPLAINT.search(q):
        return None, "complaint_answered"
    probs = reply_problems(a, item["source"], item["lang"])
    if probs:
        return None, probs[0]
    return a, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    name = args.name or latest_export("interim")
    d = export_dir("interim", name)
    ans_path = d / "answers_all.jsonl"
    if not ans_path.exists():
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(f"ankit-iiitdmj/mbg-answers-{name}", "answers_all.jsonl", repo_type="dataset"),
                    ans_path)
    answers = {r["item_id"]: r["answer"] for r in read_jsonl(ans_path)}

    stats = collections.Counter()
    splits = collections.defaultdict(list)
    for it in read_jsonl(d / "sft_items.jsonl"):
        if it["kind"] == "teacher":
            if it["item_id"] not in answers:
                stats["teacher_missing"] += 1
                continue
            target, reason = target_for(it, answers[it["item_id"]])
            stats[f"teacher:{reason}"] += 1
            if target is None:
                continue
        else:
            target = it["target"]
            stats[f"{it['kind']}:ok"] += 1
        kind = "handover" if target.startswith(HANDOVER_TAG) else "answer"
        sp = split_of(it["business_id"])
        splits[sp].append({"messages": it["messages"] + [{"role": "assistant", "content": target}],
                           "item_id": it["item_id"], "business_id": it["business_id"], "kind": kind,
                           "source_kind": it["kind"], "lang": it["lang"], "topic": it["topic"]})

    out = export_dir("processed", name)
    for sp, rows in splits.items():
        write_jsonl(out / f"{sp}.jsonl", rows)
    all_rows = [r for rows in splits.values() for r in rows]
    summary = {
        "export": name,
        "rows": {sp: len(rows) for sp, rows in sorted(splits.items())},
        "businesses": {sp: len({r["business_id"] for r in rows}) for sp, rows in sorted(splits.items())},
        "handover_share": round(sum(r["kind"] == "handover" for r in all_rows) / max(len(all_rows), 1), 3),
        "by_language": dict(collections.Counter(r["lang"] for r in all_rows).most_common()),
        "by_topic": dict(collections.Counter(r["topic"] for r in all_rows).most_common()),
        "checks": dict(sorted(stats.items())),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(f"{name}_08_sft", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.no_push:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/mbg-sft-{name}"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"SFT dataset for {name}")
    print(f"pushed -> {repo_id}  private={api.repo_info(repo_id, repo_type='dataset').private}")


if __name__ == "__main__":
    main()
