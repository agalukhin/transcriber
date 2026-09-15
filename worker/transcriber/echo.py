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
