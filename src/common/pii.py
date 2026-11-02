"""
Personal-data redaction for WhatsApp conversations.

Two different treatments:
- CUSTOMER messages: personal data is REMOVED (replaced with tags like <PHONE>). The model never needs
  a customer's number, email or name.
- BUSINESS messages: phone numbers / emails are business facts the model must learn to copy from the
  memory. They are replaced with CONSISTENT FAKE values (same real number -> same fake number within one
  business), so training keeps the "copy the number from memory" behaviour without real numbers.

Regex-based on purpose (fast, predictable). It will miss some names in free text; spot-check samples
before uploading anywhere (MASTER_PLAN Level 1, step 5).
"""
from __future__ import annotations

import hashlib
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Indian mobiles/landlines with optional +91 / 0 prefix, spaces or dashes inside; also wa.me links.
PHONE = re.compile(r"(?<![\w])(?:\+?91[\s-]?|0)?(?:\d[\s-]?){9}\d(?!\d)")
WA_ME = re.compile(r"(https?://)?wa\.me/\+?\d{8,15}", re.I)
AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
LONG_ID = re.compile(r"\b\d{11,}\b")          # order / account / transaction numbers
# Only explicit name fields ("Full name: praveen", "my name is Ravi", "naam - Sunita"). A separator is
# required, and "business/company/shop/firm name" is left alone (that's business info, not personal).
# "I am …" / Hindi "main …" are NOT treated as names (they mean "I am interested", "main aaunga" …).
NAME_FIELD = re.compile(
    r"(?i)(?<!business )(?<!company )(?<!shop )(?<!firm )(?<!store )"
    r"\b(full\s*name|my\s+name|name|mera\s+naam|naam)\s*(?::|-|=|\bis\b|\bhai\b)\s*"
    r"([A-Za-z][a-zA-Z.]+(?:\s+(?!phone\b|number\b|mobile\b|email\b|city\b|address\b)[A-Za-z][a-zA-Z.]+)?)"
)
# Last-resort catch for customer text: any Indian mobile, even glued to letters ("id9826571683").
MOBILE_ANY = re.compile(r"(?<!\d)(?:\+?91)?[6-9]\d{9}(?!\d)")
GREETING_NAME = re.compile(
    r"\b(Hi|Hello|Hey|Dear|Namaste|Namaskar)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(\s+(?:ji|sir|madam|mam|ma'am))?",
)


def _fake_phone(uid: str, real: str) -> str:
    last10 = re.sub(r"\D", "", real)[-10:]
    h = hashlib.sha256(f"{uid}|{last10}".encode()).hexdigest()
    digits = "".join(str(int(c, 16) % 10) for c in h[:9])
    return "9" + digits                                         # plausible 10-digit Indian mobile


def _fake_email(uid: str, real: str) -> str:
    h = hashlib.sha256(f"{uid}|{real.lower()}".encode()).hexdigest()[:6]
    return f"contact{h}@example-business.in"


def redact_customer(text: str, counts: dict) -> str:
    def sub(pattern, repl, s, key):
        new, n = pattern.subn(repl, s)
        counts[key] = counts.get(key, 0) + n
        return new

    text = sub(WA_ME, "<PHONE>", text, "phone")
    text = sub(EMAIL, "<EMAIL>", text, "email")
    text = sub(AADHAAR, "<ID>", text, "id")
    text = sub(PAN, "<ID>", text, "id")
    text = sub(PHONE, "<PHONE>", text, "phone")
    text = sub(MOBILE_ANY, "<PHONE>", text, "phone")
    text = sub(LONG_ID, "<ID>", text, "id")
    text = sub(NAME_FIELD, lambda m: f"{m.group(1)}: <NAME>", text, "name")
    text = sub(GREETING_NAME, lambda m: f"{m.group(1)}{m.group(3) or ''}", text, "name")
    return text


def redact_business(text: str, uid: str, counts: dict) -> str:
    def phone_repl(m):
        counts["biz_phone"] = counts.get("biz_phone", 0) + 1
        return _fake_phone(uid, m.group(0))

    def wa_repl(m):
        counts["biz_phone"] = counts.get("biz_phone", 0) + 1
        return "https://wa.me/91" + _fake_phone(uid, m.group(0))

    def email_repl(m):
        counts["biz_email"] = counts.get("biz_email", 0) + 1
        return _fake_email(uid, m.group(0))

    text = WA_ME.sub(wa_repl, text)
    text = EMAIL.sub(email_repl, text)
    text, n = AADHAAR.subn("<ID>", text); counts["id"] = counts.get("id", 0) + n
    text, n = PAN.subn("<ID>", text); counts["id"] = counts.get("id", 0) + n
    text = PHONE.sub(phone_repl, text)
    # Customer names inside business greetings ("Hello Ramesh ji") -> "Hello ji"
    text, n = GREETING_NAME.subn(lambda m: f"{m.group(1)}{m.group(3) or ''}", text)
    counts["name"] = counts.get("name", 0) + n
    return text


def pseudo_id(prefix: str, value: str) -> str:
    """Stable pseudonymous id (e.g. biz_3f9a1c2d) so exports can be linked without real ids."""
    return f"{prefix}_{hashlib.sha256(str(value).encode()).hexdigest()[:10]}"
