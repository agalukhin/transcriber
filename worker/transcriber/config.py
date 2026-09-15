from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger("transcriber")

# ~/Claude/transcriber; TRANSCRIBER_ROOT lets tests and a second checkout point elsewhere
ROOT = Path(os.environ.get("TRANSCRIBER_ROOT", Path(__file__).resolve().parents[2]))


@dataclass
class Config:
    vault_root: Path = Path("~/Documents/Obsidian Vault/Вольт воркфлоу").expanduser()
    sessions_dir: Path = ROOT / "sessions"
    raw_subdir: str = "Memory/50 · Raw/Local"
    meetings_subdir: str = "Memory/30 · Meetings"
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_silence_ms: int = 700
    speech_pad_ms: int = 100
    max_utterance_s: float = 30.0
    read_chunk_s: float = 30.0      # сколько аудио за один step(), чтобы бэклог не влезал в RAM целиком
    min_embed_s: float = 1.0
    speaker_similarity: float = 0.5
    echo_overlap: float = 0.8
    echo_similarity: float = 0.6
    echo_wait_s: float = 1.0
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    language: str = "ru"
    hallucinations: list[str] = field(default_factory=list)
    glossary: list[str] = field(default_factory=list)
    keep_audio: bool = False
    marker_every_s: int = 300


def load_config(yaml_path: Path | None = None, glossary_path: Path | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    yaml_path = yaml_path or ROOT / "worker" / "config.yaml"
    glossary_path = glossary_path or ROOT / "glossary.txt"
    data = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    data = data or {}
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        log.warning("config.yaml: unknown keys ignored: %s", ", ".join(unknown))
    kwargs = {k: v for k, v in data.items() if k in known}
    for key in ("vault_root", "sessions_dir"):
        if key in kwargs:
            kwargs[key] = Path(kwargs[key]).expanduser()
    if glossary_path.exists():
        kwargs["glossary"] = [line.strip() for line in glossary_path.read_text().splitlines() if line.strip()]
    return Config(**kwargs)
