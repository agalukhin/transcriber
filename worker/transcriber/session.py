from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Meta:
    id: str
    started_at: datetime
    app: str = "manual"
    title: str = ""
    participants: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Meta":
        return cls(id=d["id"], started_at=datetime.fromisoformat(d["started_at"]), app=d.get("app", "manual"),
                   title=d.get("title", "") or "", participants=list(d.get("participants") or []))


@dataclass
class Rec:
    """One transcribed utterance. text=None means ASR failed on it."""
    channel: str          # "mic" | "sys"
    start: float          # seconds from session start
    end: float
    text: str | None
    speaker: str          # "Me" | "Speaker N"
    emb: int | None = None  # index into State.embeddings


@dataclass
class State:
    status: str = "recording"      # recording | finalized
    mic_offset: int = 0            # samples already segmented (safe resume point)
    sys_offset: int = 0
    raw_path: str | None = None
    note_path: str | None = None
    records: list[Rec] = field(default_factory=list)
    pending_mic: list[Rec] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    speaker_of_emb: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        s = cls(**{k: v for k, v in d.items() if k not in ("records", "pending_mic")})
        s.records = [Rec(**r) for r in d.get("records", [])]
        s.pending_mic = [Rec(**r) for r in d.get("pending_mic", [])]
        return s


class Session:
    def __init__(self, dir: Path, meta: Meta, state: State):
        self.dir = Path(dir)
        self.meta = meta
        self.state = state

    @classmethod
    def load(cls, dir: Path) -> "Session":
        dir = Path(dir)
        meta = Meta.from_dict(json.loads((dir / "meta.json").read_text()))
        state_file = dir / "state.json"
        state = State.from_dict(json.loads(state_file.read_text())) if state_file.exists() else State()
        return cls(dir, meta, state)

    @property
    def mic_pcm(self) -> Path:
        return self.dir / "mic.pcm"

    @property
    def sys_pcm(self) -> Path:
        return self.dir / "sys.pcm"

    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    def save_state(self) -> None:
        atomic_write_text(self.dir / "state.json", json.dumps(asdict(self.state), ensure_ascii=False, indent=1))


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
