"""Fact check for teacher-built memories: keep only what the business really wrote.

The 14B teacher sometimes fills gaps — made-up Google Maps links, sample phone numbers, "Not specified".
Every link and every number of 4+ digits in a memory must appear in that business's own evidence;
values that fail are removed (a service keeps its name and loses only the unproven price).
"""
from __future__ import annotations

import re

from memory.prompt import memory_fill

NOT_SPECIFIED = re.compile(
    r"^\W*(not\s+(specified|provided|mentioned|available|stated|given|applicable)|n/?a|none|nil|unknown|null|-+)\W*$"
    r"|\bnot\s+(specified|provided|mentioned|stated)\b|\bprovided link\b",
    re.I,
)
TAG = re.compile(r"<(PHONE|EMAIL|ID|NAME|ADDRESS|URL)>")
URL = re.compile(r"https?://[^\s)\]>\"']+|www\.[^\s)\]>\"']+", re.I)
NUMBER = re.compile(r"\d[\d,.\-/ ]*\d")
CLOCK = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b", re.I)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


class Grounder:
    """Checks values against one business's evidence text."""

    def __init__(self, evidence_text: str):
        self.text = evidence_text
        self.digits = _digits(evidence_text)
        self.lower = evidence_text.lower()

    def ok(self, value: str) -> bool:
        if not value or NOT_SPECIFIED.search(value) or TAG.search(value):
            return False
        for u in URL.findall(value):
            if u.rstrip(".,;:!").lower() not in self.lower:
                return False
        for n in NUMBER.findall(value):
            d = _digits(n)
            if len(d) >= 4 and d not in self.digits:
                return False
        for h, mins, ap in CLOCK.findall(value):          # "10am", "8:30 pm" must be in the evidence too
            m = f"[:.]{mins}" if mins and mins != "00" else r"(?:[:.]00)?"
            if not re.search(rf"\b{h}{m}\s*{ap.lower()}\b", self.lower):
                return False
        return True


def evidence_text(ev: dict) -> str:
    return "\n".join(f"{s['customer']}\n{s['business']}" for s in ev["snippets"])


def ground_memory(memory: dict, ev_text: str) -> tuple[dict, int]:
    """(cleaned memory, number of values removed)."""
    g = Grounder(ev_text)
    removed = 0
    out: dict = {}
    for k, v in memory.items():
        if k in ("tone", "reply_language"):
            out[k] = v
        elif isinstance(v, str):
            if k == "business_name" or g.ok(v):
                out[k] = v
            else:
                removed += 1
        elif k == "services":
            items = []
            for it in v:
                if not g.ok(it["name"]):
                    removed += 1
                    continue
                svc = {"name": it["name"]}
                for f in ("price", "details"):
                    if it.get(f):
                        if g.ok(it[f]):
                            svc[f] = it[f]
                        else:
                            removed += 1
                items.append(svc)
            if items:
                out[k] = items
        elif k == "locations":
            items = []
            for it in v:
                loc = {}
                for f, val in it.items():
                    if g.ok(val):
                        loc[f] = val
                    else:
                        removed += 1
                if loc.get("address") or loc.get("maps_url"):
                    items.append(loc)
            if items:
                out[k] = items
        elif k == "faqs":
            items = [it for it in v if g.ok(it["question"]) and g.ok(it["answer"])]
            removed += len(v) - len(items)
            if items:
                out[k] = items
        elif isinstance(v, dict):                       # contact, policies
            d = {f: val for f, val in v.items() if g.ok(val)}
            removed += len(v) - len(d)
            if d:
                out[k] = d
    return out, removed


def is_good(memory: dict | None) -> bool:
    return memory_fill(memory) >= 3
