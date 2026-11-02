"""Gate 1 rules — shared by quality_audit.py and 04_build_candidates.py so both use the exact same logic."""
from __future__ import annotations

import re

# Customer said only a greeting ("hi", "hello sir", "good morning", "namaste ji" …)
GREETING = re.compile(
    r"^\W*(hi+|hii+|hello+|hel+o|hlo+|hey+|hai|namaste|namaskar|good\s*(morning|afternoon|evening|night)|"
    r"gm|hy|hlw|helo|ram\s*ram|jai\s*shree\s*ram|radhe\s*radhe|salam|assalam\w*)\W*(sir|mam|madam|ji|bhai|there)?\W*$",
    re.I,
)

# Dead-end replies: "number closed", "call us on …", "we will get back", "thank you for contacting …"
DEAD_END = re.compile(
    r"(number (is )?(closed|not in use|band)|not available|currently unavailable|out of office|"
    r"call (us )?on|contact (us )?on|call me on|whatsapp (us )?on|we will get back|will get back to you|"
    r"will contact you|our team will (call|contact)|reach us at|for (more )?(details|info).{0,20}call|"
    r"band hai|baad me|abhi busy|out of station|this number is|is no longer|thank(s| you) for (contacting|reaching|your message))",
    re.I,
)

# The "customer" message is really another business's auto-reply (bot talking to bot).
BOT_TO_BOT = re.compile(
    r"(thank(s| you) for (contacting|reaching out|your (message|enquiry|inquiry))|you'?ve reached|"
    r"outside (of )?(our )?(business|working|office) hours|we('ll| will) (get back|respond|reply)|"
    r"^\W*(hello|hi|greetings|welcome)\W+(from|to)\s+\w+|this is an automated|auto[- ]?reply|"
    r"our (team|executive) will (contact|call|get back)|please let us know how we can help)",
    re.I,
)

# Loose hint that a customer message is an enquiry (price, availability, timing, location, booking …).
# Used only to decide which STAFF-answered pairs are worth sending to the AI judge.
ENQUIRY_HINT = re.compile(
    r"\b(price|rate|cost|charges?|fees?|kitn[ae]|kitna|available|availability|stock|address|location|kaha[n]?|kahan|"
    r"timing|time|open|closed|book|booking|appointment|order|delivery|deliver|service|services|menu|size|colou?r|"
    r"discount|offer|emi|course|admission|class|demo|catalog|catalogue|details|information|info|interested|kab|"
    r"kaise|how|what|which|do you|can i|is there)\b",
    re.I,
)


def norm(t: str) -> str:
    """Normalise a reply for duplicate detection: lower-case, digits -> #, punctuation -> space."""
    t = t.lower()
    t = re.sub(r"\d", "#", t)
    t = re.sub(r"[^\w#ऀ-ॿ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def words(t: str) -> int:
    """Number of real words (2+ letters, Latin or Devanagari)."""
    return len(re.findall(r"[A-Za-zऀ-ॿ]{2,}", t))


def gate1_flags(question: str, reply: str, *, reply_repeats_in_business: int, reply_business_count: int,
                canned_per_business: int = 5, canned_global: int = 15) -> list[str]:
    """Rule flags for one question -> reply pair. Empty list = passes Gate 1."""
    f: list[str] = []
    q = question.strip()
    if GREETING.match(q):
        f.append("GREETING_Q")
    elif words(q) < 2:
        f.append("LOW_INFO_Q")
    if BOT_TO_BOT.search(q):
        f.append("BOT_TO_BOT")
    if reply_repeats_in_business >= canned_per_business or reply_business_count >= canned_global:
        f.append("CANNED")
    if DEAD_END.search(reply):
        f.append("DEAD_END")
    if len(reply) < 15 or words(reply) < 3:
        f.append("TOO_SHORT")
    if len(reply) > 1200:
        f.append("TOO_LONG")
    return f
