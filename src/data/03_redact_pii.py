"""
Level 1 · Step 3 — Redact personal data.

Input : data/interim/<export>/clean_conversations.jsonl     (from step 02)
Output: data/interim/<export>/redacted_conversations.jsonl
        data/interim/<export>/redaction_samples.txt          (for the manual spot-check)
        logs/<export>_03_redact.json

- Customer turns : phones, emails, IDs, names -> <PHONE> <EMAIL> <ID> <NAME>
- Business turns : phones/emails -> consistent FAKE values per business (see common/pii.py);
                   customer names in greetings removed.
- Business uid and chat_id -> pseudonymous ids (biz_xxxx / chat_xxxx). Real ids never leave this machine.

After running: open redaction_samples.txt and check ~200 lines by hand before anything is uploaded
(MASTER_PLAN Level 1, step 5).

Run from mbg-ai-model/:
    python src/data/03_redact_pii.py
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_report  # noqa: E402
from common.pii import pseudo_id, redact_business, redact_customer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    ap.add_argument("--samples", type=int, default=200, help="Redacted turn pairs to write for spot-checking")
    args = ap.parse_args()

    name = args.name or latest_export("interim")
    d = export_dir("interim", name)
    src = d / "clean_conversations.jsonl"
    out = d / "redacted_conversations.jsonl"

    counts: dict = {}
    c = collections.Counter()
    rng = random.Random(42)
    reservoir: list[str] = []
    seen_pairs = 0
    with out.open("w", encoding="utf-8") as f:
        for conv in read_jsonl(src):
            uid = conv["uid"]
            for t in conv["turns"]:
                red = ((lambda s: redact_customer(s, counts)) if t["role"] == "customer"
                       else (lambda s: redact_business(s, uid, counts)))
                for p in t.get("parts", []):
                    p["text"] = red(p["text"])
                t["text"] = "\n".join(p["text"] for p in t["parts"]) if t.get("parts") else red(t["text"])
            conv["business_id"] = pseudo_id("biz", uid)
            conv["conversation_id"] = pseudo_id("chat", f"{conv['export']}|{conv['chat_id']}")
            del conv["uid"], conv["chat_id"]
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            c["chats"] += 1
            # reservoir-sample answered pairs for the manual check
            turns = conv["turns"]
            for i in range(len(turns) - 1):
                if turns[i]["role"] == "customer" and turns[i + 1]["role"] == "business":
                    seen_pairs += 1
                    item = (f"[{conv['business_id']} · {','.join(turns[i + 1]['authors'])}]\n"
                            f"  CUSTOMER: {turns[i]['text'][:400]}\n"
                            f"  BUSINESS: {turns[i + 1]['text'][:400]}\n")
                    if len(reservoir) < args.samples:
                        reservoir.append(item)
                    else:
                        j = rng.randrange(seen_pairs)
                        if j < args.samples:
                            reservoir[j] = item

    (d / "redaction_samples.txt").write_text("\n".join(reservoir), encoding="utf-8")
    report = {"export": name, **c, "replacements": counts}
    write_report(f"{name}_03_redact", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}\nSpot-check: {d / 'redaction_samples.txt'}")


if __name__ == "__main__":
    main()
