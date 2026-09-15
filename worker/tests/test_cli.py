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
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(3.0)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(2.0)])))
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
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(3.0)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(2.0)])))
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
    (d / "mic.pcm").write_bytes(to_pcm_bytes(silence(3.0)))
    (d / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(2.0)])))
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
    (d2 / "mic.pcm").write_bytes(to_pcm_bytes(silence(3.0)))
    (d2 / "sys.pcm").write_bytes(to_pcm_bytes(np.concatenate([tone(1.0, 880), silence(2.0)])))
    dm.once()
    assert d2 in dm.pipes
    shutil.rmtree(d2)
    assert dm.once() is False
    assert d2 not in dm.pipes
