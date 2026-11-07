"""Fixed demo test: the same customer chats sent to each model, so replies can be compared and reviewed."""
from __future__ import annotations

from typing import Callable

from demo.chat import answer

CHATS = [
    ("greeting", ["hii"]), ("greeting", ["hlo"]), ("greeting", ["Good morning"]), ("greeting", ["namaste"]),
    ("language", ["Hindi me baat karo", "cycle ka price kya hai?"]), ("language", ["Can you talk in English?", "timings?"]),
    ("off-topic", ["icecream milegi?"]), ("off-topic", ["ek image bana do"]), ("off-topic", ["aaj ki news batao"]),
    ("off-topic", ["mera homework kar do"]), ("off-topic", ["tell me a joke"]), ("off-topic", ["laptop chahiye"]),
    ("off-topic", ["scooty ki service karte ho?"]),
    ("timing", ["dukan kab khulti hai?"]), ("timing", ["saturday ko khula hai kya?"]),
    ("location", ["shop kaha hai?"]), ("location", ["phone number do"]),
    ("product", ["gear cycle ka price batao"]), ("product", ["8 saal ke bacche ke liye cycle chahiye"]),
    ("product", ["ladies cycle hai kya?"]), ("product", ["Hero Jet Gold kitne ki hai?"]),
    ("product", ["5000 ke andar cycle milegi?"]), ("product", ["Hercules Roadeo Beast available hai?"]),
    ("product", ["tyre tube milta hai?"]), ("product", ["puncture repair karte ho?"]),
    ("policy", ["EMI milega?"]), ("policy", ["home delivery karte ho?"]), ("policy", ["warranty kitni hai?"]),
    ("policy", ["purani cycle exchange hogi?"]),
    ("not in memory", ["electric cycle hai kya?"]), ("not in memory", ["Diwali offer kya hai?"]),
    ("not in memory", ["cycle rent pe milti hai?"]),
    ("hand-over", ["cycle kharab ho gayi, complaint karni hai"]), ("hand-over", ["mujhe refund chahiye"]),
    ("hand-over", ["payment kaise karu?"]), ("hand-over", ["owner se baat karni hai"]),
    ("english", ["What cycles do you have for adults?"]), ("english", ["Do you sell helmets?"]),
    ("hindi", ["दुकान कहाँ है?"]), ("hindi", ["बच्चों की साइकिल की कीमत क्या है?"]),
    ("follow-up", ["gear cycle dikhao", "isme sabse sasti kaunsi hai?"]),
]


def run_checks(memory: dict, asks: dict[str, Callable[[list[dict]], str]]) -> list[dict]:
    """One row per model per customer message, in chat order."""
    rows = []
    for n, (kind, turns) in enumerate(CHATS):
        for model, ask in asks.items():
            history: list[dict] = []
            for turn in turns:
                r = answer(memory, turn, history, ask)
                rows.append({"chat": n, "kind": kind, "model": model, "message": turn, **r})
                history += [{"role": "user", "content": turn}, {"role": "assistant", "content": r["text"]}]
    return rows
