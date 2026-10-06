from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .talk import BOT_PATTERNS as TALK_BOT_PATTERNS, ROOM_PATTERNS as TALK_ROOM_PATTERNS

log = logging.getLogger("transcriber")

# ~/Claude/transcriber; TRANSCRIBER_ROOT lets tests and a second checkout point elsewhere
ROOT = Path(os.environ.get("TRANSCRIBER_ROOT", Path(__file__).resolve().parents[2]))

# каталог по умолчанию отдельной константой: тесты подменяют его, чтобы не читать рабочую библиотеку
VOICES_DIR = ROOT / "voices"


@dataclass
class Config:
    vault_root: Path = Path("~/Documents/Obsidian Vault/Вольт воркфлоу").expanduser()
    private_dir: Path = Path("~/Documents/Приватные записи").expanduser()   # приватные записи мимо вольта
    sessions_dir: Path = ROOT / "sessions"
    bench_dir: Path = ROOT / "bench"   # эталонные записи стенда качества и прогоны по ним
    raw_subdir: str = "Memory/50 · Raw/Local"
    meetings_subdir: str = "Memory/30 · Meetings"
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_silence_ms: int = 700
    speech_pad_ms: int = 100
    max_utterance_s: float = 30.0
    block_min_s: float = 45.0        # короче блок не закрываем: whisper любит окно ~30 с и контекст в нём
    block_max_s: float = 90.0        # длиннее не копим, иначе текст ждёт слишком долго
    block_gap_s: float = 1.5         # пауза, по которой блок режется, когда минимум набран
    block_gap_keep_s: float = 0.5    # сколько тишины оставляем в склейке от реальной паузы
    block_idle_s: float = 4.0        # тишина после последней реплики — закрываем блок, не ждём
    carry_context: bool = True       # хвост прошлого блока канала уходит в промпт следующего
    read_chunk_s: float = 30.0      # сколько аудио за один step(), чтобы бэклог не влезал в RAM целиком
    min_embed_s: float = 1.0
    speaker_similarity: float = 0.3
    voices_dir: Path = field(default_factory=lambda: VOICES_DIR)   # библиотека голосовых профилей, по файлу на человека
    voice_match_similarity: float = 0.45        # ниже — кластер остаётся «Speaker N»: ошибиться именем хуже
    voice_min_samples: int = 3                  # кластер короче — имени не даём, отпечаток ненадёжен
    # Плитки участников из интерфейса Толка (talk.jsonl рядом с дорожками сессии)
    talk_room_patterns: list[str] = field(default_factory=lambda: list(TALK_ROOM_PATTERNS))
    talk_bot_patterns: list[str] = field(default_factory=lambda: list(TALK_BOT_PATTERNS))
    talk_shared_names: list[str] = field(default_factory=list)   # общие микрофоны и владелец: имён из Толка не получают
    talk_autolearn: bool = True                 # уверенно названный Толком кластер сам идёт в библиотеку голосов
    echo_overlap: float = 0.8
    echo_similarity: float = 0.6
    echo_strong_similarity: float = 0.8   # дословный повтор — эхо даже при коротком перекрытии
    echo_wait_s: float = 1.0
    echo_suppress: bool = False   # вычитать эхо динамиков из микрофона по дорожке sys (aec.py)
    echo_beta: float = 2.0        # во сколько раз смелее вычитаем: запас на недооценку тракта
    echo_floor: float = 0.05      # ниже этого коэффициента корзину не опускаем — останется дыра
    echo_kept_max: float = 0.15   # реплика микрофона, от которой подавитель оставил меньше, — остаток эха; 0 — выкл.
    echo_kept_min_cover: float = 0.8   # ...если она хотя бы настолько накрыта речью динамиков
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    mlx_cache_mb: int = 2048     # потолок кеша Metal-буферов MLX; без него whisper за сутки съедал всю память (asr.py)
    idle_restart_mb: int = 3500  # встреч нет, а воркер держит больше — выходит, launchd поднимает чистый; 0 — выкл.
    abandoned_after_min: int = 10   # запись без STOP, чьё аудио столько не растёт, — брошена: финализируем
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
    for key in ("vault_root", "private_dir", "sessions_dir", "voices_dir", "bench_dir"):
        if key in kwargs:
            kwargs[key] = Path(kwargs[key]).expanduser()
    if glossary_path.exists():
        kwargs["glossary"] = [line.strip() for line in glossary_path.read_text().splitlines() if line.strip()]
    return Config(**kwargs)
