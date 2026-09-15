from __future__ import annotations

from pathlib import Path

import numpy as np

BYTES_PER_SAMPLE = 2  # int16 mono


class PcmReader:
    """Reads int16 mono PCM from a file that may still be growing, starting at a sample offset."""

    def __init__(self, path: Path, offset_samples: int = 0):
        self.path = Path(path)
        self.offset = int(offset_samples)

    def available(self) -> int:
        if not self.path.exists():
            return 0
        return max(0, self.path.stat().st_size // BYTES_PER_SAMPLE - self.offset)

    def read_all(self) -> np.ndarray:
        n = self.available()
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        with open(self.path, "rb") as f:
            f.seek(self.offset * BYTES_PER_SAMPLE)
            raw = f.read(n * BYTES_PER_SAMPLE)
        data = np.frombuffer(raw, dtype=np.int16)
        self.offset += len(data)
        return data.astype(np.float32) / 32768.0
