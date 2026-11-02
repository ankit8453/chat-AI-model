"""Judge prompt, verdict parser and keep rules (Gate 3)."""
from __future__ import annotations

import json
import re

TOPICS = ["price", "availability", "product_info", "timing", "location", "booking", "order_delivery",
          "support_issue", "company_info", "job", "other"]
LANGS = ["english", "hindi", "hinglish", "marathi", "other"]

SYSTEM = """You review WhatsApp chats between a BUSINESS and a person messaging it. You decide whether one
CUSTOMER message + the BUSINESS reply is good training data for a business-assistant AI.

Answer with ONE JSON object and nothing else:
{"is_customer_enquiry": true|false,
 "topic": "price|availability|product_info|timing|location|booking|order_delivery|support_issue|company_info|job|other",
 "reply_answers": true|false,
 "reply_quality": 1-5,
 "has_specific_facts": true|false,
 "unsafe_or_wrong": true|false,
 "language": "english|hindi|hinglish|marathi|other"}

Definitions
- is_customer_enquiry: true ONLY if the CUSTOMER message is a real customer or prospect asking this business
  about its products, services, prices, availability, timings, location, booking, orders, delivery,
  a problem with its service, or the company itself. FALSE for staff/internal team chat, personal or family
  chat, spam, forwarded messages, lead lists, automated/auto-reply messages, or messages with no clear request.
- reply_answers: true if the BUSINESS reply directly addresses what the customer asked (not generic,
  not evasive, not about something else).
- reply_quality: 5 = clear, correct-looking, polite, short WhatsApp-style answer; 3 = partly useful;
  1 = useless or wrong.
- has_specific_facts: true if the reply states specific prices, numbers, addresses, timings or offers.
- unsafe_or_wrong: true if the reply is rude, makes a doubtful promise, pretends to be a named person,
  replies in a different language than the customer, or looks made up.
- language: language of the CUSTOMER message (hinglish = Hindi written in English letters).

Example 1
CUSTOMER: gold ring ka rate kya hai aaj?
BUSINESS: Aaj 22K gold ka rate ₹6,950/gram hai. Ring ka price design aur weight par depend karta hai — photo bhejiye, main estimate bata dunga.
{"is_customer_enquiry": true, "topic": "price", "reply_answers": true, "reply_quality": 5, "has_specific_facts": true, "unsafe_or_wrong": false, "language": "hinglish"}

Example 2
CUSTOMER: bhai kal office kitne baje aana hai
BUSINESS: 10 baje aa jana, meeting hai
{"is_customer_enquiry": false, "topic": "other", "reply_answers": true, "reply_quality": 2, "has_specific_facts": true, "unsafe_or_wrong": false, "language": "hinglish"}"""


def build_messages(c: dict, ctx_chars: int = 400, q_chars: int = 1200, a_chars: int = 1200) -> list[dict]:
    """Chat messages for one candidate (system + user). ctx_chars=0 drops the earlier-chat context."""
    ctx = ""
    if ctx_chars > 0:
        ctx = "\n".join(f"{'CUSTOMER' if t['role'] == 'customer' else 'BUSINESS'} (earlier): {t['text'][:ctx_chars]}"
                        for t in c.get("context", []))
    user = (("EARLIER IN THE CHAT\n" + ctx + "\n\n") if ctx else "") + \
           f"CUSTOMER: {c['question'][:q_chars]}\nBUSINESS: {c['reply'][:a_chars]}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# Progressively smaller settings, tried in order until the prompt fits the model's context window.
# Devanagari (Hindi/Marathi) costs far more tokens per character than English, so char limits alone
# are not enough — the fit is checked in real tokens.
_FIT_STEPS = [(400, 1200, 1200), (200, 800, 800), (0, 700, 700), (0, 400, 400), (0, 200, 200)]


def build_messages_fitted(c: dict, count_tokens, max_input_tokens: int) -> tuple[list[dict], bool]:
    """(messages, trimmed) — messages guaranteed to be within max_input_tokens if any step fits."""
    msgs = build_messages(c)
    for i, (ctx_chars, q_chars, a_chars) in enumerate(_FIT_STEPS):
        msgs = build_messages(c, ctx_chars, q_chars, a_chars)
        if count_tokens(msgs) <= max_input_tokens:
            return msgs, i > 0
    return msgs, True


_JSON = re.compile(r"\{.*?\}", re.S)


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


def parse_verdict(text: str) -> dict | None:
    """Parse the judge's JSON. Returns None if it can't be parsed into the expected shape."""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        out = {
            "is_customer_enquiry": _as_bool(v["is_customer_enquiry"]),
            "topic": v.get("topic") if v.get("topic") in TOPICS else "other",
            "reply_answers": _as_bool(v["reply_answers"]),
            "reply_quality": max(1, min(5, int(v["reply_quality"]))),
            "has_specific_facts": _as_bool(v.get("has_specific_facts", False)),
            "unsafe_or_wrong": _as_bool(v.get("unsafe_or_wrong", False)),
            "language": v.get("language") if v.get("language") in LANGS else "other",
        }
    except (KeyError, TypeError, ValueError):
        return None
    return out


def keep_decisions(v: dict | None) -> dict:
    """keep_question: the customer message is a real enquiry (we can use the question).
    keep_pair: also the original reply is good enough to learn from (still checked against memory later)."""
    if v is None:
        return {"keep_question": False, "keep_pair": False}
    kq = v["is_customer_enquiry"]
    kp = kq and v["reply_answers"] and v["reply_quality"] >= 4 and not v["unsafe_or_wrong"]
    return {"keep_question": kq, "keep_pair": kp}
