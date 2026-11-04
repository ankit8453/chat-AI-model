"""Claim check (Level 1 step 8b): does a teacher reply say anything its memory chunks do not support?

Step 8 checks links, numbers and times mechanically, but not plain claims such as "we have Fortuner available" or
"hamare paas vacancy hai". Those taught the trial model to invent stock and services, so a model reads each reply
against the memory it was written from. Greedy, one short JSON answer per reply.
"""
from __future__ import annotations

import json
import re

CLAIM_SYSTEM = """You check replies written for a business's WhatsApp assistant. The assistant may state ONLY what is in BUSINESS INFO or in the earlier chat.

Mark the reply "unsupported" if it states or promises anything specific that is not there, for example:
- the business has, sells, offers or can do a product, service, stock, vacancy, delivery, area or treatment that is not listed
- a price, charge, distance, duration, date or quantity that is not given
- an action or promise: "we will send", "technician will come", "we have checked", "your order is confirmed"

Mark it "supported" if every specific statement is in BUSINESS INFO or the earlier chat. These are always fine: greetings, thanks, asking the customer a question, repeating what the customer said, saying what the business does in the words of BUSINESS INFO, telling the customer to call or visit using details from BUSINESS INFO.

Reply with JSON only: {"verdict": "supported" or "unsupported", "claim": "<the first unsupported statement, or empty>"}"""


def claim_messages(source: str, messages: list[dict], reply: str, count_tokens, max_input_tokens: int) -> list[dict]:
    """messages = the student's chat without the system prompt; older turns are dropped if the prompt is too long."""
    def build(turns: list[dict]) -> list[dict]:
        who = {"user": "CUSTOMER", "assistant": "BUSINESS"}
        chat = "\n".join(f"{who[m['role']]}: {m['content'][:400]}" for m in turns)
        return [{"role": "system", "content": CLAIM_SYSTEM},
                {"role": "user", "content": f"BUSINESS INFO\n{source}\n\nEARLIER CHAT\n{chat}\n\nREPLY TO CHECK\n{reply}"}]
    conv = build(messages)
    return conv if count_tokens(conv) <= max_input_tokens else build(messages[-1:])


def parse_claim(text: str | None) -> dict:
    """{"verdict": "supported" | "unsupported" | None, "claim": str}; None = could not be read."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    v = str(d.get("verdict", "")).strip().lower()
    return {"verdict": v if v in ("supported", "unsupported") else None, "claim": str(d.get("claim") or "")[:200]}
