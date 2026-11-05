"""
Live chat for the demo: strict memory search around the model, for one business.

The code hands over by itself when the customer needs a person, or when the memory has nothing on the question, so the
model is never asked in those cases. After the model replies, the reply is checked against the chunks it was given, and
a reply that claims more than they say becomes the hand-over too. Search is by words and topics; the Memory API
(Level 6) replaces it with embeddings.
"""
from __future__ import annotations

import re
from typing import Callable

from sft.build import (_COMMON, COMPLAINT, HANDOVER_DEFAULTS, HANDOVER_TAG, LEAKS, NEEDS_HUMAN, TOPIC_TYPES, _norm, _words,
                       detect_language, grounded, is_handover, makes_promise, memory_chunks, system_prompt, unsupported_terms)

HISTORY_MESSAGES = 2        # training examples carry at most the last exchange
CHUNKS_PER_REPLY = 4        # plus the business header: the most a training example carried
TOPICS = {
    "timing": r"\b(time|timing|timings|open|opening|close|closing|closed|khul\w*|band|kab|samay|sunday|ravivar)\b|समय|खुल|बंद|कब",
    "contact": r"\b(phone|mobile|number|contact|whatsapp)\b|नंबर|फोन",
    "location": r"\b(address|location|map|kaha|kahan|kidhar|pata)\b|पता|कहाँ|कहां",
    "price": r"\b(price|prices|rate|rates|cost|kitna|kitne|kitni|kimat|keemat|daam|emi|discount|offer|offers|sasta)\b"
             r"|कीमत|दाम|रेट|कितन",
    "order_delivery": r"\b(delivery|deliver|courier|shipping|ghar)\b|डिलीवरी",
    "company_info": r"\b(about|business|company|services|service|products|kya milta|kya karte|what do you)\b|सेवा",
}
COVERS = {"timing": {"timings"}, "contact": {"contact"}, "location": {"location"}, "price": {"services", "offers"},
          "company_info": {"about", "services"}}
FILLER = set("hai hain kya aap the and for you are kar koi not can sell".split())
OPENERS = re.compile(r"^\W*(hi+|hello|hey|namaste|namaskar|good (morning|afternoon|evening)|ok+|okay|thanks?|thank you|"
                     r"dhanyavaad|shukriya|haan?|ji|details?|info|jankari|batao|help)\W*$|^\W*(नमस्ते|धन्यवाद|हाँ|जी)\W*$", re.I)


def find_topic(question: str) -> str | None:
    return next((t for t, p in TOPICS.items() if re.search(p, question, re.I)), None)


def same_word(a: str, b: str) -> bool:
    """tyre = tyres, cycle = cycles."""
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def memory_covers(chunks: list[dict], question: str, topic: str | None) -> bool:
    """True if the memory has a section that can answer this topic, or shares a meaningful word with the question."""
    if topic in COVERS:
        sections = [c for c in chunks if c["type"] in COVERS[topic]]
        if topic == "price":                            # a product list without prices cannot answer a price question
            sections = [c for c in sections if re.search(r"\d", c["text"])]
        return bool(sections)
    words = _words(question) - _COMMON - FILLER
    return any(same_word(w, m) for w in words for c in chunks for m in _words(c["text"]))


def pick_chunks(chunks: list[dict], query: str, topic: str | None) -> list[dict]:
    """Header + best chunks. A word found in few chunks counts more, so "warranty" beats "cycle" in a shop's memory."""
    rest = chunks[1:]
    found = {w: [c for c in rest if any(same_word(w, m) for m in _words(c["text"]))] for w in _words(query) - _COMMON - FILLER}
    want = TOPIC_TYPES.get(topic, set())

    def score(c: dict) -> float:
        rarity = sum(1 / len(cs) for cs in found.values() if any(x is c for x in cs))
        return rarity + (0.5 if c["type"] in want else 0)

    return [chunks[0]] + sorted(rest, key=score, reverse=True)[:CHUNKS_PER_REPLY]


def checked(reply: str, question: str, source: str, handover: str) -> tuple[bool, str]:
    """(ok, reason): the reply may go to the customer only if it stays inside the chunks it was given."""
    r = reply.strip()
    low = r.lower()
    if not r or is_handover(r, handover) or HANDOVER_TAG.lower() in low or _norm(handover) in _norm(r):
        return False, "model handed over"
    if "needs_human" in low or any(w in low for w in LEAKS):
        return False, "internal text in reply"
    if not grounded(r, source):
        return False, "number or link not in memory"
    if unsupported_terms(question, r, source):
        return False, "claim not in memory"
    if makes_promise(r, source) and not re.search(r"\bhere (?:is|are)\b", r, re.I):
        return False, "promise not in memory"
    return True, "ok"


def answer(memory: dict, question: str, history: list[dict], ask: Callable[[list[dict]], str]) -> dict:
    """{'text', 'needs_human', 'reason'} for one customer message. `history` = earlier {'role','content'} messages."""
    handover = memory.get("handover_message") or HANDOVER_DEFAULTS[detect_language(question)]

    def hand_over(reason: str) -> dict:
        return {"text": handover, "needs_human": True, "reason": reason}

    if COMPLAINT.search(question) or NEEDS_HUMAN.search(question):
        return hand_over("complaint, payment or asks for a person")
    chunks = memory_chunks(memory)
    topic = find_topic(question)
    if not OPENERS.match(question) and not memory_covers(chunks, question, topic):
        return hand_over("nothing in memory")

    earlier = history[-HISTORY_MESSAGES:]
    query = " ".join([m["content"] for m in earlier if m["role"] == "user"] + [question])
    search_topic = "location" if topic == "contact" else topic or "other"
    picked = pick_chunks(chunks, query, search_topic)
    messages = [{"role": "system", "content": system_prompt(memory["business_name"], picked, handover)},
                *earlier, {"role": "user", "content": question}]
    reply = ask(messages)
    ok, reason = checked(reply, question, "\n".join(c["text"] for c in picked) + "\n" + handover, handover)
    return {"text": reply.strip(), "needs_human": False, "reason": "ok"} if ok else hand_over(reason)
