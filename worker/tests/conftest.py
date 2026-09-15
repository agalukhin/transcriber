from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

SR = 16000


def tone(seconds: float, freq: float = 440.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def to_pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def fake_prob(chunk: np.ndarray) -> float:
    """Speech = loud enough. Lets tests build speech/silence patterns from tones."""
    return 1.0 if float(np.abs(chunk).max()) > 0.05 else 0.0


class FakeAsr:
    """Returns a label derived from the dominant frequency so tests can tell utterances apart."""

    def __init__(self, fail_on: set[int] | None = None):
        self.fail_on = fail_on or set()
        self.calls: list[tuple[int, str]] = []

    def __call__(self, audio: np.ndarray, prompt: str = "") -> str:
        f = dominant_freq(audio)
        self.calls.append((f, prompt))
        if f in self.fail_on:
            raise RuntimeError("asr boom")
        return f"tone{f}"


def dominant_freq(audio: np.ndarray, sr: int = SR) -> int:
    spectrum = np.abs(np.fft.rfft(audio))
    return int(round(np.fft.rfftfreq(len(audio), 1 / sr)[int(spectrum.argmax())]))


class FakeEmbedder:
    """One-hot-ish vector per frequency band: same tone → identical embedding."""

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        f = dominant_freq(audio)
        v = np.zeros(8, dtype=np.float32)
        v[(f // 100) % 8] = 1.0
        return v


def make_session(sessions_dir: Path, session_id: str = "2026-09-15-1430", app: str = "kontur.talk",
                 title: str = "Тестовая встреча", participants: list[str] | None = None) -> Path:
    d = sessions_dir / session_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({
        "id": session_id,
        "started_at": "2026-09-15T14:30:00+03:00",
        "app": app,
        "title": title,
        "participants": participants or ["Егор Спирин"],
    }, ensure_ascii=False))
    return d


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    (root / "Memory/50 · Raw/Local").mkdir(parents=True)
    (root / "Memory/30 · Meetings").mkdir(parents=True)
    return root
