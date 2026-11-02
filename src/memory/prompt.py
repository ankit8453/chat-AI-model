"""Teacher prompt + parser for building one business memory (Level 1 step 6b) in our format."""
from __future__ import annotations

import json
import re

SYSTEM = """You build a BUSINESS MEMORY for a WhatsApp business assistant. You get real messages that ONE
business sent to its customers (with the customer message they answered, when there was one). Write down
only facts about this business that are clearly stated in those business messages.

Answer with ONE JSON object and nothing else, using only these keys (leave out any key you have no facts for):
{"business_name": "...",
 "category": "e.g. Restaurant, Salon, Clinic, Coaching, Mobile shop, Car dealer",
 "about": "2-3 short lines about what the business does",
 "services": [{"name": "...", "price": "free text, only if stated", "details": "short, only if stated"}],
 "timings": "e.g. Mon-Sat 10am-8pm, Sunday closed",
 "locations": [{"address": "...", "city": "...", "maps_url": "..."}],
 "contact": {"phone": "...", "whatsapp": "...", "email": "...", "website": "..."},
 "faqs": [{"question": "a question customers really ask", "answer": "the business's answer"}],
 "policies": {"delivery": "...", "booking": "...", "payment": "...", "refund": "...", "warranty": "..."},
 "offers": "current offers, only if stated",
 "tone": "friendly | formal | professional",
 "handover_message": "only if the business usually sends customers somewhere for details, e.g. 'For exact price please call us on 9xxxxxxxxx'"}

Rules
- Copy facts exactly as written (names, prices, numbers, addresses, links). Never guess or add facts.
- If two messages disagree (for example an old and a new price), use the message with the newest date.
- Ignore anything that is not about this business: staff/internal chat, personal chat, messages forwarded
  from other businesses, sales pitches the business received, and customer names or customer numbers.
- services: up to 25 most important items. faqs: up to 10, only common customer questions with a clear answer.
- Write every value in English, or in the language the business used for that fact. Keep values short.
- If the messages say nothing useful about the business, answer {"business_name": "UNKNOWN"}."""


def _evidence_text(ev: dict, n_snippets: int) -> str:
    lines = []
    for s in ev["snippets"][:n_snippets]:
        block = f"[{s['date']}]"
        if s["customer"]:
            block += f"\nCUSTOMER: {s['customer']}"
        block += f"\nBUSINESS: {s['business']}"
        lines.append(block)
    return "\n\n".join(lines)


def build_messages(ev: dict, n_snippets: int | None = None) -> list[dict]:
    n = len(ev["snippets"]) if n_snippets is None else n_snippets
    user = "MESSAGES FROM THE BUSINESS\n\n" + _evidence_text(ev, n) + "\n\nNow write the BUSINESS MEMORY JSON."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_messages_fitted(ev: dict, count_tokens, max_input_tokens: int) -> tuple[list[dict], int]:
    """(messages, snippets_used) — drops the lowest-ranked snippets until the prompt fits in real tokens."""
    n = len(ev["snippets"])
    msgs = build_messages(ev, n)
    while n > 1 and count_tokens(msgs) > max_input_tokens:
        n = max(1, int(n * 0.8))
        msgs = build_messages(ev, n)
    return msgs, n


# ---------------------------------------------------------------- parsing + validation (our schema v1)

_STR_KEYS = ["business_name", "category", "about", "timings", "offers", "handover_message"]
_CONTACT_KEYS = ["phone", "whatsapp", "email", "website"]
_TONES = {"friendly", "formal", "professional"}
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _s(v, limit: int = 600) -> str:
    if v is None or isinstance(v, (dict, list)):
        return ""
    return _BOLD.sub(r"\1", str(v)).strip()[:limit]


def _outer_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else None


def parse_memory(text: str) -> dict | None:
    """Parse the teacher's JSON into a clean memory dict (schema v1). None if unusable."""
    raw = _outer_json(text or "")
    if raw is None:
        return None
    try:
        m = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(m, dict):
        return None
    out: dict = {}
    for k in _STR_KEYS:
        v = _s(m.get(k))
        if v:
            out[k] = v
    if not out.get("business_name") or out["business_name"].upper() == "UNKNOWN":
        return None
    services = []
    for it in m.get("services") or []:
        if isinstance(it, dict) and _s(it.get("name")):
            svc = {"name": _s(it["name"], 120)}
            for f in ("price", "details"):
                if _s(it.get(f)):
                    svc[f] = _s(it[f], 300)
            services.append(svc)
    if services:
        out["services"] = services[:25]
    locs = []
    for it in m.get("locations") or []:
        if isinstance(it, dict):
            loc = {f: _s(it.get(f), 300) for f in ("address", "city", "maps_url") if _s(it.get(f))}
            if loc:
                locs.append(loc)
    if locs:
        out["locations"] = locs[:5]
    contact = m.get("contact") if isinstance(m.get("contact"), dict) else {}
    contact = {k: _s(contact.get(k), 200) for k in _CONTACT_KEYS if _s(contact.get(k))}
    if contact:
        out["contact"] = contact
    faqs = []
    for it in m.get("faqs") or []:
        if isinstance(it, dict) and _s(it.get("question")) and _s(it.get("answer")):
            faqs.append({"question": _s(it["question"], 200), "answer": _s(it["answer"], 500)})
    if faqs:
        out["faqs"] = faqs[:10]
    pol = m.get("policies") if isinstance(m.get("policies"), dict) else {}
    pol = {str(k): _s(v, 400) for k, v in pol.items() if _s(v)}
    if pol:
        out["policies"] = pol
    tone = _s(m.get("tone")).lower()
    out["tone"] = tone if tone in _TONES else "friendly"
    out["reply_language"] = "auto"
    return out


def memory_fill(m: dict | None) -> int:
    """How many useful sections the memory has (for the health check and filtering thin memories)."""
    if not m:
        return 0
    keys = ["about", "services", "timings", "locations", "contact", "faqs", "policies", "offers"]
    return sum(1 for k in keys if m.get(k))


def health_check(rows: list[dict]) -> None:
    """Abort after the FIRST shard of a session if the teacher output looks broken."""
    n = len(rows)
    if n == 0:
        return
    fail = sum(1 for r in rows if r["memory"] is None) / n
    avg_fill = sum(r["fill"] for r in rows) / n
    print(f"health: parse_fail={fail:.0%}  avg_sections={avg_fill:.1f}  (n={n})")
    if fail > 0.25:
        raise SystemExit(f"ABORT: {fail:.0%} of teacher outputs are unusable — check the model/prompt.")
    if avg_fill < 1.5:
        raise SystemExit(f"ABORT: memories are almost empty (avg {avg_fill:.1f} sections) — check the prompt.")
    print("health check passed")
