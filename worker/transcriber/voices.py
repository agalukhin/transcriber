"""Библиотека голосов: по одному JSON-профилю на человека.

Встречи идут с одним и тем же кругом коллег, поэтому отпечаток голоса стоит переносить
из встречи во встречу. Профиль копит сэмплы (`transcriber learn`), а на финализации
центроид кластера сверяется с центроидами профилей.

Ошибиться именем хуже, чем оставить «Speaker N», поэтому правило сопоставления
консервативное: сходство не ниже порога И отрыв от второго кандидата не меньше ``MARGIN``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .session import atomic_write_text

log = logging.getLogger("transcriber")

MAX_SAMPLES = 50     # больше полусотни отпечатков одного человека центроид уже не двигают
MARGIN = 0.05        # отрыв от второго кандидата; ближе — считаем, что голоса неразличимы


def unit(x) -> np.ndarray | None:
    """Единичный вектор или None, если считать нечего (нули, NaN, пустой ряд)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0 or not np.all(np.isfinite(a)):
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else None


def centroid_of(embeddings) -> list[float] | None:
    """Центроид — среднее единичных векторов; невалидные ряды выбрасываются."""
    units = [u for u in (unit(e) for e in embeddings) if u is not None]
    if not units:
        return None
    dim = len(units[0])
    units = [u for u in units if len(u) == dim]
    c = unit(np.mean(units, axis=0))
    return None if c is None else [float(x) for x in c]


def safe_filename(name: str) -> str:
    """Имя человека в имя файла: разделители пути и прочее запрещённое — вон."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[/\\:|#^\[\]\x00-\x1f]", " ", name)).strip(" .")
    return cleaned or "без имени"


@dataclass
class VoiceProfile:
    name: str
    count: int = 0
    centroid: list[float] = field(default_factory=list)
    samples: list[list[float]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceProfile":
        return cls(
            name=str(d["name"]),
            count=int(d.get("count") or 0),
            centroid=[float(x) for x in d.get("centroid") or []],
            samples=[[float(x) for x in s] for s in d.get("samples") or []],
            sources=[str(s) for s in d.get("sources") or []],
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "count": self.count, "centroid": self.centroid,
                "samples": self.samples, "sources": self.sources}


class VoiceLibrary:
    """Профили голосов в каталоге ``dir``; читается лениво, пишется по файлу на человека."""

    def __init__(self, dir: Path, similarity: float = 0.45, margin: float = MARGIN,
                 max_samples: int = MAX_SAMPLES):
        self.dir = Path(dir)
        self.similarity = similarity
        self.margin = margin
        self.max_samples = max_samples
        self.profiles: dict[str, VoiceProfile] = {}
        self._loaded = False

    # ---- storage ----------------------------------------------------------

    def load(self) -> "VoiceLibrary":
        """Перечитать каталог. Битый файл — предупреждение в лог, остальные профили живут."""
        self.profiles = {}
        self._loaded = True
        if not self.dir.is_dir():
            return self
        for p in sorted(self.dir.glob("*.json")):
            try:
                prof = VoiceProfile.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception as e:  # noqa: BLE001 — один битый профиль не ломает библиотеку
                log.warning("voices: %s пропущен (%s)", p.name, e)
                continue
            self.profiles[prof.name] = prof
        return self

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    def path_of(self, name: str) -> Path:
        return self.dir / f"{safe_filename(name)}.json"

    def _save(self, prof: VoiceProfile) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path_of(prof.name), json.dumps(prof.to_dict(), ensure_ascii=False, indent=1))

    # ---- api --------------------------------------------------------------

    def names(self) -> list[str]:
        self._ensure()
        return sorted(self.profiles)

    def get(self, name: str) -> VoiceProfile | None:
        self._ensure()
        return self.profiles.get(name)

    def add(self, name: str, embeddings: list[list[float]], source: str) -> VoiceProfile | None:
        """Дописать отпечатки в профиль ``name`` и пересчитать центроид. None — учить не на чем."""
        self._ensure()
        fresh = [[float(x) for x in u] for u in (unit(e) for e in embeddings) if u is not None]
        if not fresh:
            return None
        prof = self.profiles.get(name) or VoiceProfile(name=name)
        prof.count += len(fresh)
        prof.samples = (prof.samples + fresh)[-self.max_samples:]
        if source not in prof.sources:
            prof.sources.append(source)
        prof.centroid = centroid_of(prof.samples) or []
        self.profiles[name] = prof
        self._save(prof)
        return prof

    def match(self, embedding) -> tuple[str, float] | None:
        """Лучший профиль и косинусное сходство — если прошёл порог и оторвался от второго."""
        self._ensure()
        e = unit(embedding)
        if e is None:
            return None
        sims: list[tuple[float, str]] = []
        for name, prof in self.profiles.items():
            c = unit(prof.centroid)
            if c is None or len(c) != len(e):      # профиль другой размерности — не наш эмбеддер
                continue
            sims.append((float(np.dot(e, c)), name))
        if not sims:
            return None
        sims.sort(reverse=True)
        best_sim, best_name = sims[0]
        if best_sim < self.similarity:
            return None
        if len(sims) > 1 and best_sim - sims[1][0] < self.margin:
            log.info("voices: %s и %s неразличимы (%.3f vs %.3f) — оставляем Speaker",
                     best_name, sims[1][1], best_sim, sims[1][0])
            return None
        return best_name, best_sim
