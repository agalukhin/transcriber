import json
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from transcriber import cli, daemon
from transcriber.config import Config
from tests.conftest import FakeAsr, FakeEmbedder, SR, fake_prob, make_session, silence, to_pcm_bytes, tone


def write_wav(path: Path, audio: np.ndarray, sr: int = SR, channels: int = 1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        data = np.repeat(audio[:, None], channels, axis=1).reshape(-1) if channels > 1 else audio
        w.writeframes(to_pcm_bytes(data))


def fake_models():
    return daemon.Models(asr=FakeAsr(), embed=FakeEmbedder(), prob_factory=lambda: fake_prob)


def test_from_wav_runs_full_pipeline(vault, tmp_path):
    mic = tmp_path / "mic.wav"
    sys_ = tmp_path / "sys.wav"
    write_wav(mic, np.concatenate([tone(1.0, 440), silence(4.0)]))
    write_wav(sys_, np.concatenate([silence(2.0, sr=48000), tone(1.5, 880, sr=48000), silence(1.5, sr=48000)]), sr=48000, channels=2)
    cfg = Config(vault_root=vault, sessions_dir=tmp_path / "sessions")
    raw = cli.from_wav(cfg, mic, sys_, title="Из файла", app="manual", models=fake_models())
    text = raw.read_text()
    assert "Me: tone440" in text and "Speaker 1: tone880" in text and "финализирован" in text
    sdir = next((tmp_path / "sessions").iterdir())
    assert json.loads((sdir / "state.json").read_text())["status"] == "finalized"
    assert not (sdir / "mic.pcm").exists()


def test_daemon_once_processes_and_finalizes(vault, tmp_path):
    sessions = tmp_path / "sessions"
    d = make_session(sessions)
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(5.5)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(4.5)])))
    cfg = Config(vault_root=vault, sessions_dir=sessions)
    dm = daemon.Daemon(cfg, models=fake_models())
    assert dm.once() is True
    assert json.loads((d / "state.json").read_text())["status"] == "recording"
    (d / "STOP").touch()
    dm.once()
    assert json.loads((d / "state.json").read_text())["status"] == "finalized"
    assert dm.once() is False        # finalized sessions are skipped


def write_wav_width(path: Path, audio: np.ndarray, width: int, sr: int = SR):
    a = np.clip(audio, -1, 1)
    if width == 1:
        data = np.round(a * 127 + 128).astype(np.uint8).tobytes()
    elif width == 2:
        data = (a * 32767).astype("<i2").tobytes()
    elif width == 3:
        i32 = np.round(a * (2 ** 23 - 1)).astype("<i4")
        data = np.frombuffer(i32.tobytes(), dtype=np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        i32 = np.round(a * (2 ** 31 - 1)).astype("<i4")
        data = i32.tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(data)


def test_read_wav_supports_8_24_and_32_bit(tmp_path):
    audio = tone(0.5, 440)
    ref = cli.read_wav(_w(tmp_path, audio, 2), SR)
    for width, tol in ((1, 0.02), (3, 1e-4), (4, 1e-4)):
        got = cli.read_wav(_w(tmp_path, audio, width), SR)
        assert len(got) == len(ref)
        assert float(np.abs(got - ref).max()) < tol


def _w(tmp_path: Path, audio: np.ndarray, width: int) -> Path:
    p = tmp_path / f"w{width}.wav"
    write_wav_width(p, audio, width)
    return p


def test_from_wav_reports_unsupported_width_as_value_error(tmp_path, vault, capsys):
    import pytest

    mic = _wav_with_bits(tmp_path / "mic.wav", bits=40, frames=10)
    with pytest.raises(ValueError):
        cli.read_wav(mic, SR)
    cfg = Config(vault_root=vault, sessions_dir=tmp_path / "sessions")
    import transcriber.cli as climod

    monkey_cfg = cfg
    orig = climod.load_config
    climod.load_config = lambda *a, **k: monkey_cfg
    try:
        with pytest.raises(SystemExit) as e:
            cli.main(["from-wav", str(mic), str(mic)])
        assert e.value.code == 2
    finally:
        climod.load_config = orig
    assert "40-bit" in capsys.readouterr().err


def _wav_with_bits(path: Path, bits: int, frames: int, sr: int = SR) -> Path:
    """Модуль wave такой файл не запишет — собираем RIFF руками."""
    import struct

    width = bits // 8
    data = b"\x00" * (frames * width)
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sr, sr * width, width, bits)
    body = b"WAVE" + fmt + struct.pack("<4sI", b"data", len(data)) + data
    path.write_bytes(struct.pack("<4sI", b"RIFF", len(body)) + body)
    return path


def test_from_wav_does_not_reuse_session_dir(vault, tmp_path):
    mic = tmp_path / "mic.wav"
    sys_ = tmp_path / "sys.wav"
    write_wav(mic, silence(1.0))
    write_wav(sys_, silence(1.0))
    cfg = Config(vault_root=vault, sessions_dir=tmp_path / "sessions")
    started = datetime(2026, 9, 15, 14, 30, 0).astimezone()
    cli.from_wav(cfg, mic, sys_, title="Раз", app="manual", models=fake_models(), started_at=started)
    cli.from_wav(cfg, mic, sys_, title="Два", app="manual", models=fake_models(), started_at=started)
    dirs = sorted(p.name for p in (tmp_path / "sessions").iterdir())
    assert dirs == ["2026-09-15-143000", "2026-09-15-143000-1"]


def test_daemon_backs_off_broken_session_and_keeps_others(vault, tmp_path, caplog):
    sessions = tmp_path / "sessions"
    broken = sessions / "0000-broken"
    broken.mkdir(parents=True)
    (broken / "meta.json").write_text("{ это не json")
    d = make_session(sessions)
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(5.5)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(4.5)])))
    cfg = Config(vault_root=vault, sessions_dir=sessions)
    dm = daemon.Daemon(cfg, models=fake_models())
    for _ in range(5):
        dm.once()
    assert json.loads((d / "state.json").read_text())["records"]      # здоровая сессия обработана
    assert dm.failures[broken] == 5
    caplog.clear()
    with caplog.at_level("DEBUG"):
        for _ in range(3):
            dm.once()
    assert dm.failures[broken] == 5                                   # больше не трогаем
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_daemon_forgets_vanished_and_externally_finalized_sessions(vault, tmp_path):
    import shutil

    sessions = tmp_path / "sessions"
    d = make_session(sessions)
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(5.5)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(4.5)])))
    cfg = Config(vault_root=vault, sessions_dir=sessions)
    dm = daemon.Daemon(cfg, models=fake_models())
    dm.once()
    assert d in dm.pipes
    state = json.loads((d / "state.json").read_text())
    state["status"] = "finalized"                     # финализировали снаружи, через cli finalize
    (d / "state.json").write_text(json.dumps(state))
    assert dm.once() is False
    assert d not in dm.pipes

    d2 = make_session(sessions, session_id="2026-09-15-1500", title="Вторая")
    (d2 / "mic.pcm").write_bytes(to_pcm_bytes(silence(5.5)))
    (d2 / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(4.5)])))
    dm.once()
    assert d2 in dm.pipes
    shutil.rmtree(d2)
    assert dm.once() is False
    assert d2 not in dm.pipes


def test_daemon_creates_vault_files_before_loading_models(vault, tmp_path, monkeypatch, caplog):
    """Клик «открыть заметку» в приложении не должен ждать 10 секунд загрузки моделей."""
    sessions = tmp_path / "sessions"
    d = make_session(sessions)
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(5.5)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(4.5)])))
    cfg = Config(vault_root=vault, sessions_dir=sessions)

    def boom(_cfg):
        raise RuntimeError("модели не загрузились")

    monkeypatch.setattr(daemon, "load_models", boom)
    dm = daemon.Daemon(cfg)
    with caplog.at_level("ERROR", logger="transcriber"):
        assert dm.once() is False            # без моделей работы не сделано
    state = json.loads((d / "state.json").read_text())
    assert Path(state["raw_path"]).exists() and Path(state["note_path"]).exists()
    assert "Статус: идёт запись" in Path(state["raw_path"]).read_text()
    assert dm.failures[d] == 1               # падение ушло в обычный путь обработки, без краша


def test_daemon_does_not_rewrite_vault_files_for_a_resumed_session(vault, tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    d = make_session(sessions)
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(5.5)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(4.5)])))
    cfg = Config(vault_root=vault, sessions_dir=sessions)
    daemon.Daemon(cfg, models=fake_models()).once()
    raw = Path(json.loads((d / "state.json").read_text())["raw_path"])
    raw.write_text("правка снаружи")
    monkeypatch.setattr(daemon, "load_models", lambda _cfg: (_ for _ in ()).throw(RuntimeError("no models")))
    dm = daemon.Daemon(cfg)
    dm.once()
    assert raw.read_text() == "правка снаружи"   # raw_path уже есть — заглушку не перезаписываем


SYNTHETIC_RECORDS = [
    {"channel": "sys", "start": 0.0, "end": 2.0, "text": "Давайте обсудим контракт", "speaker": "Speaker 1", "emb": 0},
    {"channel": "sys", "start": 4.0, "end": 6.0, "text": "Термины, Саша Галухин.", "speaker": "Speaker 2", "emb": 1},
    {"channel": "sys", "start": 8.0, "end": 10.0, "text": "Мы отдадим SQL", "speaker": "Speaker 3", "emb": 2},
    {"channel": "sys", "start": 12.0, "end": 14.0, "text": "Аааа... Ааа...", "speaker": "Speaker 4", "emb": 3},
    {"channel": "sys", "start": 16.0, "end": 18.0, "text": "Ну да, конечно", "speaker": "Speaker 5", "emb": 4},
    {"channel": "mic", "start": 7.0, "end": 11.0, "text": "Мы отдадим SQL.", "speaker": "Me", "emb": None},
    {"channel": "mic", "start": 20.0, "end": 22.0, "text": "Я думаю, надо начать с октября", "speaker": "Me", "emb": None},
    {"channel": "mic", "start": 30.0, "end": 32.0, "text": "Владислав, Магнит, Омни, CDP, Катя Галухин.", "speaker": "Me", "emb": None},
]
SYNTHETIC_EMB = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0.25, 0, 0.97]]


def _synthetic_session(sessions: Path) -> Path:
    d = make_session(sessions, title="Встреча")
    (d / "state.json").write_text(json.dumps({
        "status": "finalized", "mic_offset": 0, "sys_offset": 0, "raw_path": None, "note_path": None,
        "records": SYNTHETIC_RECORDS, "pending_mic": [],
        "embeddings": SYNTHETIC_EMB, "speaker_of_emb": [0, 1, 2, 3, 4], "errors": [],
    }, ensure_ascii=False))
    return d


def test_refinalize_cleans_texts_echo_and_speakers_without_audio(vault, tmp_path, caplog):
    sessions = tmp_path / "sessions"
    d = _synthetic_session(sessions)
    cfg = Config(vault_root=vault, sessions_dir=sessions,
                 glossary=["Магнит", "Омни", "CDP", "Саша Галухин", "Катя Гладникова"])
    with caplog.at_level("INFO", logger="transcriber"):
        raw = cli.refinalize(cfg, "2026-09-15-1430")

    state = json.loads((d / "state.json").read_text())
    assert state["speaker_of_emb"] == [0, 1, 0, 1, 0]         # пять «спикеров» схлопнулись в двух
    kinds = [(r["channel"], r["text"]) for r in state["records"]]
    assert ("mic", "Мы отдадим SQL.") not in kinds            # эхо убрано
    assert ("mic", "Я думаю, надо начать с октября") in kinds
    assert [r["text"] for r in state["records"] if r["channel"] == "sys"] == [
        "Давайте обсудим контракт", "", "Мы отдадим SQL", "", "Ну да, конечно"]
    assert [r["speaker"] for r in state["records"] if r["channel"] == "sys"] == [
        "Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2", "Speaker 1"]

    body = raw.read_text().split("## Полный транскрипт", 1)[1]
    assert "Статус: финализирован повторно" in raw.read_text()
    assert "Галухин" not in body and "Аааа" not in body
    assert body.count("Мы отдадим SQL") == 1
    assert "prompt leak" in caplog.text and "filler" in caplog.text


def test_refinalize_is_idempotent(vault, tmp_path):
    sessions = tmp_path / "sessions"
    d = _synthetic_session(sessions)
    cfg = Config(vault_root=vault, sessions_dir=sessions, glossary=["Магнит", "Омни", "CDP", "Саша Галухин"])
    first = cli.refinalize(cfg, "2026-09-15-1430").read_text()
    again = cli.refinalize(cfg, "2026-09-15-1430")
    assert again.read_text().replace("повторно", "") == first.replace("повторно", "")
    assert json.loads((d / "state.json").read_text())["speaker_of_emb"] == [0, 1, 0, 1, 0]


def test_refinalize_command_prints_raw_path(vault, tmp_path, capsys, monkeypatch):
    sessions = tmp_path / "sessions"
    _synthetic_session(sessions)
    cfg = Config(vault_root=vault, sessions_dir=sessions, glossary=["Магнит"])
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    cli.main(["refinalize", "2026-09-15-1430"])
    assert capsys.readouterr().out.strip().endswith("Local-2026-09-15-1430-Встреча.md")


def _session_with_audio(sessions: Path, vault: Path) -> tuple[Path, Path, Path]:
    """Сессия с сохранённым аудио и устаревшим state.json от прошлого прогона."""
    d = make_session(sessions)
    (d / "mic.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 440), silence(5.0)])))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([silence(2.0), tone(1.5, 880), silence(2.0)])))
    raw = vault / "Memory/50 · Raw/Local/Local-2026-09-15-1430-Старое-имя.md"
    note = vault / "Memory/30 · Meetings/MTG-2026-09-15 1430 · Старое имя.md"
    raw.write_text("прошлый прогон")
    note.write_text("заглушка")
    (d / "state.json").write_text(json.dumps({
        "status": "finalized", "mic_offset": 10 ** 7, "sys_offset": 10 ** 7,
        "raw_path": str(raw), "note_path": str(note),
        "records": [{"channel": "sys", "start": 0.0, "end": 1.0, "text": "старый мусор",
                     "speaker": "Speaker 1", "emb": 0}],
        "pending_mic": [{"channel": "mic", "start": 0.0, "end": 1.0, "text": "хвост",
                         "speaker": "Me", "emb": None}],
        "embeddings": [[1.0, 0.0, 0.0]], "speaker_of_emb": [0], "errors": ["старая ошибка"],
    }, ensure_ascii=False))
    return d, raw, note


def test_retranscribe_reruns_pipeline_over_saved_audio(vault, tmp_path):
    sessions = tmp_path / "sessions"
    d, raw, note = _session_with_audio(sessions, vault)
    cfg = Config(vault_root=vault, sessions_dir=sessions, keep_audio=True)
    out = cli.retranscribe(cfg, "2026-09-15-1430", models=fake_models())
    state = json.loads((d / "state.json").read_text())
    assert out == raw and state["raw_path"] == str(raw) and state["note_path"] == str(note)
    assert state["status"] == "finalized" and state["errors"] == []
    assert {r["text"] for r in state["records"]} == {"tone440", "tone880"}
    assert state["pending_mic"] == [] and state["mic_offset"] < 10 ** 7
    text = raw.read_text()
    assert "Me: tone440" in text and "Speaker 1: tone880" in text and "старый мусор" not in text
    assert (d / "mic.pcm").exists()                      # keep_audio: аудио остаётся под следующий прогон


def test_retranscribe_drops_audio_when_keep_audio_is_off(vault, tmp_path):
    sessions = tmp_path / "sessions"
    d, _raw, _note = _session_with_audio(sessions, vault)
    cli.retranscribe(Config(vault_root=vault, sessions_dir=sessions), "2026-09-15-1430", models=fake_models())
    assert not (d / "mic.pcm").exists() and not (d / "sys.pcm").exists()


def test_retranscribe_command_prints_path_and_refuses_without_audio(vault, tmp_path, capsys, monkeypatch):
    import pytest

    sessions = tmp_path / "sessions"
    d, raw, _note = _session_with_audio(sessions, vault)
    cfg = Config(vault_root=vault, sessions_dir=sessions, keep_audio=True)
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(cli, "load_models", lambda _cfg: fake_models())
    cli.main(["retranscribe", "2026-09-15-1430"])
    assert capsys.readouterr().out.strip() == str(raw)

    (d / "mic.pcm").unlink()
    (d / "sys.pcm").unlink()
    with pytest.raises(SystemExit) as e:
        cli.main(["retranscribe", "2026-09-15-1430"])
    assert e.value.code == 2
    assert "аудио" in capsys.readouterr().err
