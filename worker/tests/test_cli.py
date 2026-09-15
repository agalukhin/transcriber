import json
import wave
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
