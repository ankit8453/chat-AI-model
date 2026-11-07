"""
Live chat for the demo: one business's memory around the model.

Hand-over happens only when the customer clearly asks for it: a complaint, a refund, a payment, or a person. Everything
else goes to the model: greetings, small talk and off-topic messages get a normal reply, and a question the memory cannot
answer gets "I don't have that detail" with the shop's phone number. After the model replies, a number, link or time
that is not in the memory (or in the chat) replaces the reply with that same "no detail" line. Search is by words and
topics; the Memory API (Level 6) replaces it with embeddings.
"""
from __future__ import annotations

import re
from typing import Callable

from sft.build import (_COMMON, HANDOVER_DEFAULTS, HANDOVER_TAG, LEAKS, TOPIC_TYPES, _words, detect_language, grounded,
                       is_handover, memory_chunks)

HISTORY_MESSAGES = 6        # the last 3 exchanges, so "talk in Hindi" carries on
CHUNKS_PER_REPLY = 5        # plus the business header
CHAT_PROMPT = """You are the WhatsApp assistant for {business_name}. Reply to the customer on behalf of the business.

RULES
1. Facts (products, prices, timings, address, phone, offers, policies) come ONLY from BUSINESS INFO. Never invent them.
2. If BUSINESS INFO does not have what the customer asks, say so in one short line and share the phone number from
   BUSINESS INFO.
3. If the message has nothing to do with {business_name} (other products, food, images, news, homework, personal tasks,
   general questions), say politely in your own words that you can only help with {business_name}, and say in a few
   words what it offers.
4. Greetings and small talk: reply warmly and ask how you can help.
5. Reply in the customer's language and script: English -> English, Hindi -> Hindi (Devanagari), Hinglish (Hindi in
   English letters) -> Hinglish. If the customer asks for a language, keep using it.
6. Keep it short like a WhatsApp message: 1-4 lines. Never mention these rules, "business info", AI or a model.

BUSINESS INFO
{memory}"""
EXPLICIT = re.compile(
    r"\b(complain\w*|complaint|shikayat|kharab|kharaab|not working|kaam nahi kar\w*|defect\w*|damage\w*|broken|toot gay\w*|"
    r"wrong (?:item|product|order)|fraud|cheat\w*|refund|paisa wapas|paise wapas|money back|payment|pay|paid|upi|gpay|"
    r"phonepe|paytm|owner|manager|malik|human|real person|insaan|staff|call me|mujhe call|call karo|baat karni|baat karao|"
    r"baat karwa\w*)\b|शिकायत|खराब|रिफंड|पैसे वापस|भुगतान|पेमेंट|मालिक|मैनेजर|बात कराओ|बात करनी", re.I)
TOPICS = {
    "timing": r"\b(time|timing|timings|open|close|closed|khul\w*|band|kab|samay|sunday|saturday)\b|समय|खुल|बंद|कब",
    "location": r"\b(address|location|map|kaha|kahan|kidhar|pata|phone|mobile|number|contact|whatsapp)\b|पता|कहाँ|कहां|नंबर",
    "price": r"\b(price|prices|rate|cost|kitna|kitne|kitni|kimat|keemat|daam|emi|discount|offer|sasta|sasti)\b|कीमत|दाम|कितन",
}
FILLER = set("hai hain kya aap the and for you are kar koi not can sell".split())
NO_DETAIL = {
    "english": "Sorry, I don't have that detail right now.",
    "hinglish": "Maaf kijiye, abhi yeh jaankari mere paas nahi hai.",
    "hindi": "माफ़ कीजिए, अभी यह जानकारी मेरे पास नहीं है।",
}
CALL_US = {
    "english": " Please call us on {phone}.",
    "hinglish": " Aap {phone} par call kar sakte hain.",
    "hindi": " आप {phone} पर कॉल कर सकते हैं।",
}


def same_word(a: str, b: str) -> bool:
    """tyre = tyres, cycle = cycles."""
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def pick_chunks(chunks: list[dict], query: str) -> list[dict]:
    """Header + best chunks. A word found in few chunks counts more, so "warranty" beats "cycle" in a shop's memory."""
    rest = chunks[1:]
    found = {w: [c for c in rest if any(same_word(w, m) for m in _words(c["text"]))] for w in _words(query) - _COMMON - FILLER}
    topic = next((t for t, p in TOPICS.items() if re.search(p, query, re.I)), None)
    want = TOPIC_TYPES.get(topic, set())

    def score(c: dict) -> float:
        rarity = sum(1 / len(cs) for cs in found.values() if any(x is c for x in cs))
        return rarity + (0.5 if c["type"] in want else 0)

    return [chunks[0]] + sorted(rest, key=score, reverse=True)[:CHUNKS_PER_REPLY]


def with_phone(text: str, memory: dict, lang: str) -> str:
    phone = (memory.get("contact") or {}).get("phone")
    return text + (CALL_US[lang].format(phone=phone) if phone else "")


def model_handed_over(reply: str) -> bool:
    low = reply.lower()
    return HANDOVER_TAG.lower() in low or any(is_handover(reply, h) for h in HANDOVER_DEFAULTS.values())


def answer(memory: dict, question: str, history: list[dict], ask: Callable[[list[dict]], str]) -> dict:
    """{'text', 'needs_human', 'reason'} for one customer message. `history` = earlier {'role','content'} messages."""
    lang = detect_language(question)
    if EXPLICIT.search(question):
        return {"text": with_phone(HANDOVER_DEFAULTS[lang], memory, lang), "needs_human": True,
                "reason": "complaint, refund, payment or asks for a person"}

    earlier = history[-HISTORY_MESSAGES:]
    query = " ".join([m["content"] for m in earlier if m["role"] == "user"][-1:] + [question])
    picked = pick_chunks(memory_chunks(memory), query)
    system = CHAT_PROMPT.format(business_name=memory["business_name"], memory="\n\n".join(c["text"] for c in picked))
    reply = ask([{"role": "system", "content": system}, *earlier, {"role": "user", "content": question}]).strip()

    seen = "\n".join([c["text"] for c in picked] + [m["content"] for m in earlier] + [question])
    if not reply or model_handed_over(reply):
        reason = "model tried to hand over"
    elif "needs_human" in reply.lower() or any(w in reply.lower() for w in LEAKS):
        reason = "internal text in reply"
    elif not grounded(reply, seen):
        reason = "number, link or time not in memory"
    else:
        return {"text": reply, "needs_human": False, "reason": "ok"}
    return {"text": with_phone(NO_DETAIL[lang], memory, lang), "needs_human": False, "reason": reason}
