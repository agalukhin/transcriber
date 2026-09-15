from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
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
        started = datetime.fromisoformat(d["started_at"])
        if started.tzinfo is None:                 # writer omitted the offset — treat as local time
            started = started.astimezone()
        return cls(id=d["id"], started_at=started, app=d.get("app", "manual"),
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

    @classmethod
    def from_dict(cls, d: dict) -> "Rec":
        return cls(**_known(cls, d))


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
    speaker_names: dict[str, str] = field(default_factory=dict)   # индекс кластера (строкой) → имя из библиотеки голосов
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        s = cls(**{k: v for k, v in _known(cls, d).items() if k not in ("records", "pending_mic")})
        s.records = [Rec.from_dict(r) for r in d.get("records") or []]
        s.pending_mic = [Rec.from_dict(r) for r in d.get("pending_mic") or []]
        return s


def _known(cls, d: dict) -> dict:
    """Ignore keys a newer (or older) version wrote — state.json must never block a restart."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


class Session:
    def __init__(self, path: Path, meta: Meta, state: State):
        self.dir = Path(path)
        self.meta = meta
        self.state = state

    @classmethod
    def load(cls, path: Path) -> "Session":
        path = Path(path)
        meta = Meta.from_dict(json.loads((path / "meta.json").read_text()))
        state_file = path / "state.json"
        state = State.from_dict(json.loads(state_file.read_text())) if state_file.exists() else State()
        return cls(path, meta, state)

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
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")   # another process must not clobber our tmp
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())                                 # without fsync a power loss leaves an empty file
    os.replace(tmp, path)
