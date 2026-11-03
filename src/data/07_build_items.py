"""
Level 1 · Step 7 — Build the fine-tuning items (memory + real customer question -> ideal reply).

Input : data/interim/<export>/judged_all.jsonl, candidates.jsonl, business_evidence.jsonl, memories_all.jsonl
        (judge + memories are downloaded from the private HF repos if missing)
Output: data/interim/<export>/sft_items.jsonl          (push with 05_push_candidates.py --file sft_items.jsonl)
        logs/<export>_07_items.json

Each item is what the student model will see in production — system prompt with the business's retrieved
memory chunks + hand-over message, the last chat turns, the customer's message — plus how its target reply
is made:
  teacher  : written by the open teacher model (notebook 03_generate_answers), checked in step 8
  human    : the business's own staff reply, when it passes every check (facts grounded in the chunks, no
             customer name, no placeholder tags, right language) — OFF by default (USE_HUMAN_REPLIES)
  handover : the chunks that would answer the question are removed, so the right reply is the hand-over
AI/bot replies are never used as targets (third-party model outputs) — those questions go to the teacher.

Gate 4 fixes applied here: memories fact-checked against the business's own messages (memory/clean.py),
**bold** -> WhatsApp *bold*, replies with tags / customer names dropped, at most 40 questions per business,
at most 150 copies of the same question across businesses, questions with passwords / OTPs dropped.

Run from mbg-ai-model/:   python src/data/07_build_items.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import export_dir, latest_export, read_jsonl, write_jsonl, write_report  # noqa: E402
from common.quality import norm  # noqa: E402
from memory.clean import evidence_text, ground_memory, is_good  # noqa: E402
from sft.build import (HANDOVER_DEFAULTS, HANDOVER_TAG, bold_to_whatsapp, detect_language,  # noqa: E402
                       history_messages, memory_chunks, reply_problems, select_chunks, system_prompt)

PER_BUSINESS = 40
SAME_QUESTION_MAX = 150
HANDOVER_SHARE = 0.12
BUSINESS_HANDOVER_P = 0.7          # else the default hand-over, so the model learns both
# Staff replies often state facts that are not in the memory (dates, "come on the 20th"), which would teach
# the model to make things up. Off by default: those questions get a teacher reply instead.
USE_HUMAN_REPLIES = False
PRICE_WORDS = re.compile(r"\b(price|rate|cost|quotation|charges?)\b", re.I)
SECRET = re.compile(r"\b(password|passwd|pass\s?word|otp|cvv|upi\s?pin|atm\s?pin|mpin)\b", re.I)

# What to remove so that a question of this topic can no longer be answered -> hand-over is the right reply.
HANDOVER_TOPICS = {
    "price": lambda c: c["type"] in ("services", "offers") or (c["type"] == "faq" and re.search(r"\d", c["text"])),
    "availability": lambda c: c["type"] in ("services", "faq"),
    "timing": lambda c: c["type"] == "timings" or (c["type"] == "faq" and re.search(r"\d", c["text"])),
    "location": lambda c: c["type"] == "location" or (c["type"] == "faq" and re.search(r"address|location|map", c["text"], re.I)),
    "order_delivery": lambda c: c["type"] in ("policies", "faq"),
}


def fetch(d: Path, name: str, repo: str, fname: str) -> Path:
    p = d / fname
    if not p.exists():
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(repo, fname, repo_type="dataset"), p)
    return p


def pick_questions(rows: list[dict], rng: random.Random) -> list[dict]:
    """Round-robin over topics so each business contributes varied questions."""
    by_topic = collections.defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    for v in by_topic.values():
        rng.shuffle(v)
    out, topics = [], sorted(by_topic)
    while len(out) < PER_BUSINESS and any(by_topic.values()):
        for t in topics:
            if by_topic[t] and len(out) < PER_BUSINESS:
                out.append(by_topic[t].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Export name (default: latest in data/interim/)")
    args = ap.parse_args()
    name = args.name or latest_export("interim")
    d = export_dir("interim", name)
    fetch(d, name, f"ankit-iiitdmj/mbg-judge-{name}", "judged_all.jsonl")
    fetch(d, name, f"ankit-iiitdmj/mbg-memories-{name}", "memories_all.jsonl")
    stats = collections.Counter()

    # 1. fact-checked memories
    ev = {r["business_id"]: r for r in read_jsonl(d / "business_evidence.jsonl")}
    memories = {}
    for r in read_jsonl(d / "memories_all.jsonl"):
        if r["memory"] is None:
            continue
        m, removed = ground_memory(r["memory"], evidence_text(ev[r["business_id"]]))
        stats["memory_values_removed"] += removed
        if is_good(m):
            memories[r["business_id"]] = m
    print(f"good memories after fact check: {len(memories):,}")

    # 2. real questions of those businesses
    verdicts = {r["pair_id"]: r for r in read_jsonl(d / "judged_all.jsonl")
                if r["keep_question"] and r["business_id"] in memories}
    q_count = collections.Counter()
    per_biz = collections.defaultdict(list)
    for c in read_jsonl(d / "candidates.jsonl"):
        v = verdicts.get(c["pair_id"])
        if v is None:
            continue
        q = c["question"].strip()
        if SECRET.search(q):
            stats["drop_secret"] += 1
            continue
        if len(q) > 700:
            stats["drop_long_question"] += 1
            continue
        k = norm(q)[:120]
        q_count[k] += 1
        if q_count[k] > SAME_QUESTION_MAX:
            stats["drop_repeated_question"] += 1
            continue
        per_biz[c["business_id"]].append({**c, "topic": v["verdict"]["topic"], "keep_pair": v["keep_pair"],
                                          "judge_language": v["verdict"]["language"]})

    # 3. items
    items = []
    for b in sorted(per_biz):
        m = memories[b]
        chunks = memory_chunks(m)
        brng = random.Random(int(hashlib.md5(b.encode()).hexdigest()[:8], 16))
        seen = set()
        for c in pick_questions(per_biz[b], brng):
            k = norm(c["question"])
            if k in seen:
                stats["drop_duplicate_in_business"] += 1
                continue
            seen.add(k)
            rng = random.Random(int(hashlib.md5(c["pair_id"].encode()).hexdigest()[:8], 16))
            lang = detect_language(c["question"])
            if lang == "english" and c["judge_language"] in ("hinglish", "marathi"):
                lang = "hinglish"           # our word list misses many Hinglish messages; the judge reads them
            own = m.get("handover_message", "")
            if own and PRICE_WORDS.search(own) and c["topic"] != "price":
                own = ""                    # "For exact price call …" reads wrong for a location question
            handover = own if own and rng.random() < BUSINESS_HANDOVER_P else HANDOVER_DEFAULTS[lang]
            picked = select_chunks(chunks, c["question"], c["topic"], rng)
            kind, target = "teacher", None

            human = bold_to_whatsapp(c["reply"]).strip()
            source = "\n\n".join(x["text"] for x in picked) + "\n\n" + handover
            if USE_HUMAN_REPLIES and c["author"] in ("staff", "agent") and c["keep_pair"]:
                probs = reply_problems(human, source, lang)
                if not probs:
                    kind, target = "human", human
                else:
                    stats.update(f"human_rejected:{p}" for p in probs)

            if kind == "teacher" and c["topic"] in HANDOVER_TOPICS and rng.random() < HANDOVER_SHARE / 0.6:
                drop = HANDOVER_TOPICS[c["topic"]]
                picked = [chunks[0]] + [x for x in chunks[1:] if not drop(x)]
                picked = picked[:1] + rng.sample(picked[1:], min(3, len(picked) - 1))
                kind, target = "handover", f"{HANDOVER_TAG}\n{handover}"
                source = "\n\n".join(x["text"] for x in picked) + "\n\n" + handover

            messages = [{"role": "system", "content": system_prompt(m["business_name"], picked, handover)}]
            messages += history_messages(c.get("context", []))
            messages.append({"role": "user", "content": c["question"]})
            items.append({"item_id": c["pair_id"], "business_id": b, "kind": kind, "lang": lang,
                          "topic": c["topic"], "author": c["author"], "handover": handover,
                          "source": source, "messages": messages, "target": target})
            stats[f"kind:{kind}"] += 1
            stats[f"lang:{lang}"] += 1

    out = d / "sft_items.jsonl"
    write_jsonl(out, items)
    report = {"export": name, "good_memories": len(memories), "businesses_with_items": len(per_biz),
              "items": len(items), **dict(sorted(stats.items()))}
    p = write_report(f"{name}_07_items", report)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"wrote {out} · report {p}")


if __name__ == "__main__":
    main()
