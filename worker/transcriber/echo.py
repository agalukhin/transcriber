from __future__ import annotations

import re
from difflib import SequenceMatcher

from .session import Rec


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE).split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def overlap_fraction(m: Rec, others: list[Rec]) -> float:
    dur = m.end - m.start
    if dur <= 0:
        return 0.0
    total = sum(max(0.0, min(m.end, o.end) - max(m.start, o.start)) for o in others)
    return min(1.0, total / dur)


def is_echo(m: Rec, others: list[Rec], overlap: float, sim: float, strong: float = 0.8) -> bool:
    """A mic utterance is an echo of the speakers if it overlaps sys speech and says the same thing.

    Two rules, either of them is enough:
    * ``overlap`` of the mic utterance is covered by sys speech **and** the texts are ``sim``-close;
    * the texts are ``strong``-close and there is any time overlap at all — the mic side of an echo
      is often segmented into one long utterance around a short sys one, which sinks the coverage.
    """
    if not m.text:
        return False
    touching = [o for o in others if o.text and min(m.end, o.end) - max(m.start, o.start) > 0]
    if not touching:
        return False
    best = max(similarity(m.text, o.text) for o in touching)
    return (overlap_fraction(m, touching) >= overlap and best >= sim) or best >= strong


# Дежурные фразы, которые whisper сочиняет на обрывках задавленного эха (0.2–1 с остатка).
STOCK_PHRASES = {"спасибо", "спасибо за просмотр", "спасибо за внимание", "продолжение следует",
                 "всем пока", "пока", "до свидания", "угу", "ага", "да"}
RESIDUE_MAX_S = 2.0
RESIDUE_MARGIN_S = 2.0   # остаток эха всплывает и в паузах между фразами собеседника


def is_residue(m: Rec, others: list[Rec]) -> bool:
    """Короткая дежурная фраза на микрофоне, пока из динамиков кто-то говорит, — остаток эха.

    Та же фраза в тишине динамиков — реплика владельца и остаётся. Содержательный текст поверх
    чужой речи тоже остаётся: это настоящий разговор внахлёст.
    """
    if not m.text or m.end - m.start > RESIDUE_MAX_S:
        return False
    if normalize(m.text) not in STOCK_PHRASES:
        return False
    return any(o.text and min(m.end, o.end + RESIDUE_MARGIN_S) - max(m.start, o.start - RESIDUE_MARGIN_S) > 0
               for o in others)
