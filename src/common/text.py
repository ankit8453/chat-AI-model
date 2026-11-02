"""Text helpers: noise detection and light normalisation for WhatsApp messages."""
from __future__ import annotations

import re
import unicodedata

# Platform / system messages that must never become training data.
SYSTEM_NOISE = [
    re.compile(p, re.I)
    for p in [
        r"AI reply is temporarily unavailable",
        r"temporarily unavailable due to platform limits",
        r"^\s*error\s*$",
        r"this message was deleted",
        r"message deleted",
        r"^\s*<media omitted>\s*$",
        r"waiting for this message",
    ]
]

_WS = re.compile(r"[ \t ]+")
_MANY_NL = re.compile(r"\n{3,}")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MANY_NL.sub("\n\n", text)
    return text.strip()


def is_system_noise(text: str) -> bool:
    return any(p.search(text) for p in SYSTEM_NOISE)


def has_letters(text: str) -> bool:
    """True if the text has at least one letter in any script (rules out emoji/punctuation/number-only)."""
    return any(unicodedata.category(ch).startswith("L") for ch in text)


DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
# Common Hinglish words (Hindi written in Latin script) — a cheap language hint, not a classifier.
HINGLISH_HINTS = re.compile(
    r"\b(kya|hai|hain|kaise|kitna|kitne|kab|kahan|nahi|nahin|haan|ji|aap|apka|aapka|mujhe|chahiye|"
    r"karo|kar|kijiye|bhai|accha|acha|theek|thik|bhejo|batao|bataiye|rate|wala|wali|hoga|milega)\b",
    re.I,
)


def language_hint(text: str) -> str:
    """'hindi' | 'hinglish' | 'english' | 'other' — used for balancing and language-match checks."""
    if DEVANAGARI.search(text):
        return "hindi"
    if LATIN.search(text):
        return "hinglish" if HINGLISH_HINTS.search(text) else "english"
    return "other"
