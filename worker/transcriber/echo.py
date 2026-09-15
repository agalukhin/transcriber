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


def is_echo(m: Rec, others: list[Rec], overlap: float, sim: float) -> bool:
    """A mic utterance is an echo of the speakers if it overlaps sys speech and says the same thing."""
    if not m.text:
        return False
    touching = [o for o in others if o.text and min(m.end, o.end) - max(m.start, o.start) > 0]
    if overlap_fraction(m, touching) < overlap:
        return False
    return max((similarity(m.text, o.text) for o in touching), default=0.0) >= sim
