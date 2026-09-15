"""Guards against whisper output that is not speech.

Everything here is a pure function over the decoded text: the first real meeting showed that
whisper, fed an ``initial_prompt`` with participants and glossary, recites that prompt back on
unclear audio («Владислав, Магнит, Омни, CDP, Факт, Катя Галухин.»), loops n-grams
(«Магнит, Офлайн, Магнит, Офлайн…»), drops an English line into a Russian call and fills pauses
with «Аааа... Ааа...». Such an utterance is dropped from the transcript, not marked unrecognised:
there was nothing to recognise.
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
FILLER_RE = re.compile(r"^(а+|э+|м+|ну|угу|ага|хм+|ммм+|эм+)$")
PROMPT_KEYWORDS = ("Участники", "Термины")

LEAK_MIN_WORDS = 3
LEAK_SHARE = 0.6
REPEAT_TIMES = 3
VARIETY_MIN_WORDS = 6
VARIETY_SHARE = 0.35
CYRILLIC_SHARE = 0.3
FOREIGN_MIN_WORDS = 3


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def prompt_vocabulary(participants: list[str], glossary: list[str]) -> set[str]:
    """Every word whisper could echo back from the prompt built by ``asr.build_prompt``."""
    out: set[str] = set()
    for s in (*PROMPT_KEYWORDS, "Владислав", *participants, *glossary):
        out.update(tokenize(s))
    return out


def prompt_leak(text: str, prompt_words: set[str]) -> bool:
    w = tokenize(text)
    if len(w) < LEAK_MIN_WORDS:
        return False
    return sum(1 for x in w if x in prompt_words) / len(w) >= LEAK_SHARE


def _max_consecutive(w: list[str], n: int) -> int:
    best = 1
    for i in range(len(w) - n + 1):
        gram, k, j = w[i:i + n], 1, i + n
        while w[j:j + n] == gram:
            k += 1
            j += n
        best = max(best, k)
    return best


def repetitive(text: str) -> bool:
    w = tokenize(text)
    if not w:
        return False
    if any(_max_consecutive(w, n) >= REPEAT_TIMES for n in (1, 2, 3)):
        return True
    return len(w) >= VARIETY_MIN_WORDS and len(set(w)) / len(w) < VARIETY_SHARE


def foreign(text: str, language: str = "ru") -> bool:
    """For a Russian call: a line that is essentially not Cyrillic at all."""
    if language != "ru" or len(tokenize(text)) < FOREIGN_MIN_WORDS:
        return False
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    cyrillic = sum(1 for c in alpha if "Ѐ" <= c <= "ӿ")
    return cyrillic / len(alpha) < CYRILLIC_SHARE


def filler_only(text: str) -> bool:
    w = tokenize(text)
    return bool(w) and all(FILLER_RE.match(x) for x in w)


def junk_reason(text: str, prompt_words: set[str], language: str = "ru") -> str | None:
    """Name of the rule that rejects this utterance, or None if it is worth keeping."""
    if not text:
        return None
    if prompt_leak(text, prompt_words):
        return "prompt leak"
    if repetitive(text):
        return "repetitive"
    if foreign(text, language):
        return "foreign"
    if filler_only(text):
        return "filler"
    return None
