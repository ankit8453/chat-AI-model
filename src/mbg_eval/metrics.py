"""Golden-set scoring (Level 2 / 4): the same mechanical exam for the base model and every fine-tuned version.

A reply passes when
  - the hand-over decision matches the reference (hand over when the memory has no answer / a complaint,
    answer when it does), and
  - a hand-over is exactly "[HANDOVER]\\n<hand-over message>", or
  - an answer has only facts from its memory chunks (links, 3+ digit numbers, clock times), no placeholder
    tags, no customer name, the customer's script, <= 700 chars, no word loops, no mention of the rules.
"""
from __future__ import annotations

import collections

from sft.build import HANDOVER_TAG, LEAKS, is_handover, reply_problems


def score(row: dict, pred: str | None) -> dict:
    p = (pred or "").strip()
    exp_ho = row["reference"].startswith(HANDOVER_TAG)
    pred_ho = bool(p) and is_handover(p, row["handover"])
    if pred_ho:
        problems = [] if p == f"{HANDOVER_TAG}\n{row['handover']}" else ["handover_format"]
    else:
        problems = reply_problems(p, row["source"], row["lang"])
        if "handover" in p.lower():
            problems.append("handover_mid_text")
        if any(w in p.lower() for w in LEAKS):
            problems.append("mentions_rules")
    return {"item_id": row["item_id"], "expected_handover": exp_ho, "pred_handover": pred_ho,
            "decision_ok": exp_ho == pred_ho, "problems": problems,
            "passed": exp_ho == pred_ho and not problems, "chars": len(p)}


def summarize(scores: list[dict]) -> dict:
    n = len(scores) or 1
    exp_ho = [s for s in scores if s["expected_handover"]]
    exp_ans = [s for s in scores if not s["expected_handover"]]
    answered = [s for s in scores if not s["pred_handover"]]
    handed = [s for s in scores if s["pred_handover"]]
    rate = lambda xs, f: round(sum(1 for x in xs if f(x)) / max(len(xs), 1), 3)  # noqa: E731
    return {
        "n": len(scores),
        "pass_rate": rate(scores, lambda s: s["passed"]),
        "decision_accuracy": rate(scores, lambda s: s["decision_ok"]),
        "handover_recall": rate(exp_ho, lambda s: s["pred_handover"]),          # hands over when it should
        "false_handover_rate": rate(exp_ans, lambda s: s["pred_handover"]),     # hands over when it could answer
        "handover_format_ok": rate(handed, lambda s: "handover_format" not in s["problems"]),
        "grounded_rate": rate(answered, lambda s: "ungrounded_fact" not in s["problems"]),
        "language_ok": rate(answered, lambda s: "language_mismatch" not in s["problems"]),
        "uses_name": rate(answered, lambda s: "uses_name" in s["problems"]),
        "avg_chars": round(sum(s["chars"] for s in scores) / n),
        "problems": dict(collections.Counter(p for s in scores for p in s["problems"]).most_common()),
    }


def by_group(rows: list[dict], scores: list[dict], key: str) -> dict:
    """pass_rate per language / topic / ..."""
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r, s in zip(rows, scores):
        groups[r[key]].append(s)
    return {k: {"n": len(v), "pass_rate": round(sum(s["passed"] for s in v) / len(v), 3)}
            for k, v in sorted(groups.items())}


def compare_table(results: dict[str, dict]) -> str:
    """results = {"baseline": summary, "trial": summary, ...} -> printable table."""
    keys = ["pass_rate", "decision_accuracy", "handover_recall", "false_handover_rate", "handover_format_ok",
            "grounded_rate", "language_ok", "uses_name", "avg_chars"]
    names = list(results)
    lines = [f"{'metric':<22}" + "".join(f"{n:>14}" for n in names)]
    for k in keys:
        lines.append(f"{k:<22}" + "".join(f"{results[n][k]:>14}" for n in names))
    return "\n".join(lines)
