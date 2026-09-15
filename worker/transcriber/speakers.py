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
        self.last: int = 0

    @classmethod
    def from_history(cls, similarity: float, embeddings: list[list[float]], labels: list[int]) -> "OnlineClusterer":
        c = cls(similarity)
        agg: dict[int, tuple[np.ndarray, int]] = {}
        for e, l in zip(embeddings, labels):
            u = _unit(np.asarray(e))
            if l in agg:
                s, n = agg[l]
                agg[l] = (s + u, n + 1)
            else:
                agg[l] = (u, 1)
        if agg:
            ordered = sorted(agg)
            assert ordered == list(range(len(ordered))), "from_history: labels must be contiguous 0..k-1"
            for l in ordered:
                s, n = agg[l]
                c.sums.append(s)
                c.counts.append(n)
        return c

    @property
    def n_speakers(self) -> int:
        return len(self.sums)

    def assign(self, emb: np.ndarray) -> int:
        raw = np.asarray(emb, dtype=np.float32)
        if not np.all(np.isfinite(raw)) or float(np.linalg.norm(raw)) == 0.0:
            return self.last
        e = _unit(raw)
        if self.sums:
            sims = [float(np.dot(e, _unit(s))) for s in self.sums]
            sims = [s if np.isfinite(s) else -1.0 for s in sims]
            best = int(np.argmax(sims))
            if sims[best] >= self.similarity:
                self.sums[best] = self.sums[best] + e
                self.counts[best] += 1
                self.last = best
                return best
        self.sums.append(e.copy())
        self.counts.append(1)
        self.last = len(self.sums) - 1
        return self.last


def _relabel(labels: list[int]) -> list[int]:
    """Renumber cluster ids 0..k-1 in order of first appearance."""
    order: dict[int, int] = {}
    return [order.setdefault(int(l), len(order)) for l in labels]


def absorb_small_clusters(labels: list[int], embeddings, min_size: int = 2,
                          min_similarity: float = 0.1) -> list[int]:
    """Merge clusters below ``min_size`` into the nearest normal-sized one by centroid cosine.

    Single-utterance clusters are almost always whisper junk or a badly cut segment, not a person:
    on the first real meeting they were a hallucinated English line and a stretch of «Аааа...».
    A tiny cluster is kept as its own speaker when even the nearest centroid is below
    ``min_similarity``, and when there is no cluster of ``min_size`` or more to merge into.
    Result is relabelled by first appearance.
    """
    labels = [int(l) for l in labels]
    arr = np.asarray(embeddings, dtype=np.float64)
    if not labels or len(arr) != len(labels):
        return _relabel(labels)
    counts: dict[int, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    big = [l for l in counts if counts[l] >= min_size]
    if not big:
        return _relabel(labels)
    centroids = {l: _unit(np.sum([_unit(arr[i]) for i, x in enumerate(labels) if x == l], axis=0))
                 for l in counts}
    mapping: dict[int, int] = {}
    for l in counts:
        if counts[l] >= min_size:
            mapping[l] = l
            continue
        sims = [(float(np.dot(centroids[l], centroids[b])), b) for b in big]
        sims = [(s if np.isfinite(s) else -1.0, b) for s, b in sims]
        best_sim, best = max(sims)
        mapping[l] = best if best_sim >= min_similarity else l
    return _relabel([mapping[l] for l in labels])


def recluster(embeddings: np.ndarray, similarity: float) -> list[int]:
    """Average-linkage agglomerative clustering on cosine distance; labels ordered by first appearance.

    Rows that are non-finite or have zero norm are masked out of the clustering (scipy's
    ``linkage`` raises on a non-finite condensed distance matrix) and given the label of the
    previous valid row in original order, or 0 if there is none before them.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    arr = np.asarray(embeddings, dtype=np.float64)
    finite = np.isfinite(arr).all(axis=1)
    norms = np.linalg.norm(np.where(finite[:, None], arr, 0.0), axis=1)
    valid = finite & (norms > 0)
    valid_idx = np.nonzero(valid)[0]

    if len(valid_idx) == 0:
        return [0] * n
    if len(valid_idx) == 1:
        valid_labels = [0]
    else:
        x = np.stack([_unit(embeddings[i]) for i in valid_idx])
        z = linkage(x, method="average", metric="cosine")
        raw = fcluster(z, t=1.0 - similarity, criterion="distance")
        valid_labels = _relabel([int(c) for c in raw])

    labels = [0] * n
    last = 0
    vi = 0
    for i in range(n):
        if vi < len(valid_idx) and valid_idx[vi] == i:
            last = valid_labels[vi]
            vi += 1
        labels[i] = last
    return labels


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
