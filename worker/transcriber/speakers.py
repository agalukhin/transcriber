from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def label(idx: int) -> str:
    return f"Speaker {idx + 1}"


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


class OnlineClusterer:
    """Assigns each embedding to the nearest centroid (cosine) or opens a new speaker."""

    def __init__(self, similarity: float):
        self.similarity = similarity
        self.sums: list[np.ndarray] = []
        self.counts: list[int] = []

    @classmethod
    def from_history(cls, similarity: float, embeddings: list[list[float]], labels: list[int]) -> "OnlineClusterer":
        c = cls(similarity)
        for e, l in zip(embeddings, labels):
            while len(c.sums) <= l:
                c.sums.append(np.zeros(len(e), dtype=np.float32))
                c.counts.append(0)
            c.sums[l] = c.sums[l] + _unit(np.asarray(e))
            c.counts[l] += 1
        return c

    @property
    def n_speakers(self) -> int:
        return len(self.sums)

    def assign(self, emb: np.ndarray) -> int:
        e = _unit(emb)
        if self.sums:
            sims = [float(np.dot(e, _unit(s))) for s in self.sums]
            best = int(np.argmax(sims))
            if sims[best] >= self.similarity:
                self.sums[best] = self.sums[best] + e
                self.counts[best] += 1
                return best
        self.sums.append(e.copy())
        self.counts.append(1)
        return len(self.sums) - 1


def recluster(embeddings: np.ndarray, similarity: float) -> list[int]:
    """Average-linkage agglomerative clustering on cosine distance; labels ordered by first appearance."""
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    x = np.stack([_unit(e) for e in embeddings])
    z = linkage(x, method="average", metric="cosine")
    raw = fcluster(z, t=1.0 - similarity, criterion="distance")
    order: dict[int, int] = {}
    return [order.setdefault(int(c), len(order)) for c in raw]


class Embedder:
    """pyannote speaker embedding; returns a 1-D float32 vector."""

    def __init__(self, model_name: str, sample_rate: int = 16000):
        import torch
        from pyannote.audio import Inference, Model

        self.torch = torch
        self.sr = sample_rate
        model = Model.from_pretrained(model_name)
        self.inference = Inference(model, window="whole")

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        wave = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        return np.asarray(self.inference({"waveform": wave, "sample_rate": self.sr}), dtype=np.float32).reshape(-1)
