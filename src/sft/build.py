"""Training-example helpers (Level 1 step 7): memory -> chunks, chunk selection, prompts, answer checks.

The student model always sees exactly what it will see in production: the system prompt from
prompts/system_prompt.md with the business's retrieved memory chunks + hand-over message, the last turns
of the chat, and the customer's message.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from common.text import language_hint
from memory.clean import CLOCK, TAG, URL, _digits

ROOT = Path(__file__).resolve().parents[2]
HANDOVER_DEFAULTS = {k: v for k, v in json.loads((ROOT / "config/handover_messages.json").read_text(encoding="utf-8")).items()
                     if not k.startswith("_")}
HANDOVER_TAG = "[HANDOVER]"


def load_system_template() -> str:
    md = (ROOT / "prompts/system_prompt.md").read_text(encoding="utf-8")
    m = re.search(r"```text\n(.*?)\n```", md, re.S)
    if not m:
        raise SystemExit("prompts/system_prompt.md has no ```text block")
    return m.group(1)


SYSTEM_TEMPLATE = load_system_template()

# ---------------------------------------------------------------- memory -> chunks (same shape as live RAG)


def memory_chunks(m: dict) -> list[dict]:
    """[{type, text}] — the header chunk first. handover_message is not a chunk (it has its own prompt slot)."""
    head = f"{m['business_name']}" + (f" — {m['category']}" if m.get("category") else "")
    if m.get("about"):
        head += f"\n{m['about']}"
    chunks = [{"type": "about", "text": head}]
    svcs = m.get("services") or []
    for i in range(0, len(svcs), 6):
        lines = []
        for s in svcs[i:i + 6]:
            line = f"- {s['name']}"
            if s.get("price"):
                line += f": {s['price']}"
            if s.get("details"):
                line += f" ({s['details']})"
            lines.append(line)
        chunks.append({"type": "services", "text": "Products / services:\n" + "\n".join(lines)})
    if m.get("timings"):
        chunks.append({"type": "timings", "text": f"Timings: {m['timings']}"})
    for loc in m.get("locations") or []:
        parts = [loc.get("address", ""), loc.get("city", "")]
        t = "Address: " + ", ".join(p for p in parts if p)
        if loc.get("maps_url"):
            t += f"\nMap: {loc['maps_url']}"
        chunks.append({"type": "location", "text": t})
    if m.get("contact"):
        c = m["contact"]
        chunks.append({"type": "contact", "text": "Contact: " + ", ".join(f"{k.capitalize()} {v}" for k, v in c.items())})
    for f in m.get("faqs") or []:
        chunks.append({"type": "faq", "text": f"Q: {f['question']}\nA: {f['answer']}"})
    if m.get("policies"):
        chunks.append({"type": "policies", "text": "Policies:\n" + "\n".join(f"- {k}: {v}" for k, v in m["policies"].items())})
    if m.get("offers"):
        chunks.append({"type": "offers", "text": f"Current offers: {m['offers']}"})
    return chunks


TOPIC_TYPES = {
    "price": {"services", "offers", "faq"}, "availability": {"services", "faq"}, "product_info": {"services", "faq"},
    "timing": {"timings", "faq"}, "location": {"location", "contact"}, "booking": {"contact", "timings", "faq", "policies"},
    "order_delivery": {"policies", "contact", "faq"}, "support_issue": {"contact", "policies", "faq"},
    "company_info": {"about", "services", "contact"}, "job": {"contact", "faq"}, "other": {"services", "contact"},
}
_WORD = re.compile(r"[a-zऀ-ॿ]{3,}")


def _words(t: str) -> set[str]:
    return set(_WORD.findall(t.lower()))


def select_chunks(chunks: list[dict], question: str, topic: str, rng: random.Random, k: int = 3,
                  distractor_p: float = 0.35) -> list[dict]:
    """Header + top-k chunks by word overlap (+ topic boost), sometimes + 1 unrelated chunk of the same business."""
    q = _words(question)
    want = TOPIC_TYPES.get(topic, set())
    rest = chunks[1:]
    scored = sorted(rest, key=lambda c: (len(q & _words(c["text"])) + (2 if c["type"] in want else 0), rng.random()),
                    reverse=True)
    picked = scored[:k]
    others = scored[k:]
    if others and rng.random() < distractor_p:
        picked.insert(rng.randrange(len(picked) + 1), rng.choice(others))
    return [chunks[0]] + picked


# ---------------------------------------------------------------- prompts

def detect_language(text: str) -> str:
    lang = language_hint(text)
    return lang if lang in HANDOVER_DEFAULTS else "english"


def system_prompt(business_name: str, chunks: list[dict], handover: str) -> str:
    return (SYSTEM_TEMPLATE.replace("{business_name}", business_name)
            .replace("{retrieved_memory_chunks}", "\n\n".join(c["text"] for c in chunks))
            .replace("{handover_message}", handover))


TEACHER_GUIDE = """
YOU ARE WRITING THE IDEAL REPLY (it trains a smaller assistant), so follow the RULES above exactly:
- Answer the customer's actual question in the first line, using only facts from BUSINESS INFO.
- If the exact fact they need (e.g. the price of that item, stock, a date) is not in BUSINESS INFO, hand over:
  write [HANDOVER] on the first line and the HANDOVER MESSAGE on the second line, nothing else.
- If the message is vague ("more info", "details"), give a 2-3 line summary from BUSINESS INFO and ask one
  short follow-up question.
- Same language and script as the customer: Hinglish in English letters, Hindi in Devanagari, English in English.
- Never use the customer's name. At most about 60 words. *bold* only for 1-2 key words.
- Output only the reply text."""


def teacher_messages(item: dict, count_tokens, max_input_tokens: int) -> list[dict]:
    """The student's prompt + the teacher guide; drops history / trims the question if it is too long."""
    msgs = item["messages"]
    sys_msg = {"role": "system", "content": msgs[0]["content"] + "\n" + TEACHER_GUIDE}
    full = [sys_msg] + msgs[1:]
    if count_tokens(full) <= max_input_tokens:
        return full
    short = [sys_msg, msgs[-1]]
    if count_tokens(short) <= max_input_tokens:
        return short
    return [sys_msg, {"role": "user", "content": msgs[-1]["content"][:300]}]


def bold_to_whatsapp(t: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", t)


def history_messages(context: list[dict], max_chars: int = 400) -> list[dict]:
    msgs: list[dict] = []
    for t in context:
        role = "user" if t["role"] == "customer" else "assistant"
        text = bold_to_whatsapp(t["text"][:max_chars]).strip()
        if not text:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + text
        else:
            msgs.append({"role": role, "content": text})
    return msgs


# ---------------------------------------------------------------- answer checks

LEAKS = ("business info", "as an ai", "language model", "handover message", "rules")


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9ऀ-ॿ]+", " ", t.lower()).strip()


def is_handover(a: str, handover: str) -> bool:
    """[HANDOVER] and its typos ([AHANDOVER], [VHANDOVER]), or the hand-over message alone (also in brackets)."""
    if re.match(r"\W*[A-Z]?HANDOVER\b", a, re.I):
        return True
    na, nh = _norm(a), _norm(handover)
    return bool(nh) and (na in nh or nh in na) and len(na) >= 0.6 * len(nh) and len(na) <= len(nh) + 10


# Capitalised words that are followed by "ji" but are not names ("Haan ji", "Bilkul ji", "Dhanyavaad ji").
_NOT_NAMES = ("(?:Sir|Mam|Madam|Bhai|Bhaiya|Didi|Aap|Hum|Team|Haan|Han|Haa|Hanji|Theek|Thik|Bilkul|Accha|Acha|"
              "Achha|Namaste|Namaskar|Dhanyavaad|Dhanyawad|Shukriya|Sahi|Okay|Sure|Yes|Thanks|Zaroor|Jarur|"
              "Jaroor|Welcome|Bataiye|Boliye|Kahiye|Pranam|Ram|Jai)\\b")
ADDRESSES_NAME = re.compile(
    r"^\W*(?i:hi|hello|hey|dear|thanks|thank you|ok|okay|sure|ji)\W+(?!" + _NOT_NAMES + r"|There\b|All\b|Welcome\b)"
    r"[A-Z][a-z]{2,}\b|\b(?!" + _NOT_NAMES + r")[A-Z][a-z]{2,} ji\b"
)
NUM3 = re.compile(r"\d[\d,]*\d")
REPEAT = re.compile(r"\b(\w+)(?:\W+\1\b){3,}", re.I)          # same word 4+ times in a row (a generation loop)
FOREIGN_SCRIPT = re.compile(r"[؀-ۿঀ-৿぀-ヿ一-鿿가-힯]")  # Arabic/Bengali/CJK

# Customer messages in Marathi / Bengali / Gujarati. The 14B teacher writes these badly, so their answers are not
# used as targets (their hand-overs are kept).
OTHER_LANG_DEV = re.compile(r"(आहे|आहेत|मला|तुम्ही|तुम्हाला|आणि|नाही|साठी|वरती|आपण|कोणत्या|झाले|पाहिजे|मिळेल)")
OTHER_LANG_LAT = (   # only words Hindi / Hinglish do not use
    re.compile(r"\b(ahe|aahe|ahet|mala|tumhi|tumhala|majhya|majha|mazha|mazya|aani|kiti|pahije|sanga|yeil|zhala|"
               r"karto|ghya|nako)\b", re.I),
    re.compile(r"\b(ami|apni|apnar|apnara|kichu|korbo|korte|janaben|hobe|amar|tomar|bolun|korchi|porbe)\b", re.I),
    re.compile(r"\b(tame|tamne|tamaru|tamara|mane|maru|chhe|che|nathi|joye|joiye|joie|kem|hoy|puchi|aapo|karvu)\b", re.I),
)


def other_language(text: str) -> bool:
    return bool(OTHER_LANG_DEV.search(text)) or any(len(r.findall(text)) >= 2 for r in OTHER_LANG_LAT)


def grounded(answer: str, source: str) -> bool:
    """Every link, every number of 3+ digits and every clock time in the answer appears in the source text."""
    low = source.lower()
    src_digits = _digits(source)
    for u in URL.findall(answer):
        if u.rstrip(".,;:!*").lower() not in low:
            return False
    for n in NUM3.findall(answer):
        d = _digits(n)
        if len(d) >= 3 and d not in src_digits:
            return False
    for h, mins, ap in CLOCK.findall(answer):
        m = f"[:.]{mins}" if mins and mins != "00" else r"(?:[:.]00)?"
        if not re.search(rf"\b{h}{m}\s*{ap.lower()}\b", low):
            return False
    return True


# Common words that say nothing about what the business sells; anything else repeated from the customer's message
# in a positive reply, but missing from the memory, is a claim the memory does not back ("Fortuner available hai").
_COMMON = set("""what when where which about have your with this that from please want need send share details price
cost much many kitna kitne kitni kaha kahan hain karna karte chahiye available aapke aapka apke apka hamare humare
hello thanks thank sure more info information like would could should there their them they will shall been some also
only just very good okay right madam nahi kaise kaun hota hoga hogi milega milegi mujhe mera meri chahte bataiye batao
tell know give make time today tomorrow number contact call whatsapp service services product products specific
looking interested help assist location address office shop store""".split())
_TERM = re.compile(r"[a-zऀ-ॿ]{4,}")
_POSITIVE = re.compile(r"\b(available|we have|we offer|we provide|we can|we do|yes|hamare paas|humare paas|provide kar|"
                       r"mil jayega|milega|kar sakte)\b|उपलब्ध|हमारे पास|हाँ|हां", re.I)
_NEGATIVE = re.compile(r"\b(not|don't|do not|no|nahi|nhi|sorry)\b|नहीं", re.I)
PROMISE = re.compile(r"\b(we can arrange|we can update|we will|we'll|i will|i'll|let me|here (?:is|are)|we also have|"
                     r"kar dunga|kar denge|bhej dunga|bhej denge|batayenge|dekh lenge)\b", re.I)


def unsupported_terms(question: str, reply: str, source: str) -> list[str]:
    """Words of the customer's message that a positive reply repeats although the memory does not have them."""
    if not _POSITIVE.search(reply) or _NEGATIVE.search(reply):
        return []
    low_s, low_r = source.lower(), reply.lower()
    return sorted({w for w in _TERM.findall(question.lower()) if w not in _COMMON and w in low_r and w not in low_s})


def makes_promise(reply: str, source: str) -> bool:
    m = PROMISE.search(reply)
    return bool(m) and m.group(0).lower() not in source.lower()


def reply_problems(reply: str, source: str, lang: str) -> list[str]:
    """Reasons a reply must not become a training target (empty list = ok)."""
    p = []
    if TAG.search(reply):
        p.append("placeholder_tag")
    if not grounded(reply, source):
        p.append("ungrounded_fact")
    low = source.lower()
    for mt in ADDRESSES_NAME.finditer(reply):
        word = re.search(r"[A-Z][a-z]{2,}(?= ji\b)|[A-Z][a-z]{2,}\b$", mt.group(0))
        if word and word.group(0).lower() not in low:      # a word from the memory (e.g. the business name) is fine
            p.append("uses_name")
            break
    if len(reply) > 700:
        p.append("too_long")
    if REPEAT.search(reply):
        p.append("repetitive")
    if FOREIGN_SCRIPT.search(reply):
        p.append("foreign_script")
    if len(reply.strip()) < 2:
        p.append("empty")
    has_dev = bool(re.search(r"[ऀ-ॿ]", reply))
    if lang == "hindi" and not has_dev:
        p.append("language_mismatch")
    if lang in ("english", "hinglish") and has_dev:
        p.append("language_mismatch")
    return p
