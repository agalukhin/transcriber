from pathlib import Path

import numpy as np
from transcriber.config import Config
from transcriber.pipeline import Pipeline
from transcriber.session import Session
from tests.conftest import FakeAsr, FakeEmbedder, SR, fake_prob, make_session, silence, to_pcm_bytes, tone


def write(d: Path, name: str, audio: np.ndarray):
    (d / name).write_bytes(to_pcm_bytes(audio))


def build(vault, sessions, session_id="2026-09-15-1430", asr=None, **cfg_kw):
    cfg = Config(vault_root=vault, sessions_dir=sessions, **cfg_kw)
    s = Session.load(sessions / session_id)
    p = Pipeline(cfg, s, asr=asr or FakeAsr(), embed=FakeEmbedder(), prob_factory=lambda: fake_prob)
    return cfg, s, p


def test_live_step_writes_raw_and_labels(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    mic = np.concatenate([silence(0.5), tone(1.0, 440), silence(5.0)])
    sys_ = np.concatenate([silence(2.0), tone(1.5, 880), silence(1.0), tone(1.5, 1200), silence(1.5)])
    write(d, "mic.pcm", mic)
    write(d, "sys.pcm", sys_)
    cfg, s, p = build(vault, tmp_path / "sessions")
    assert p.step() is True
    texts = [(r.speaker, r.text) for r in sorted(s.state.records, key=lambda r: r.start)]
    assert texts == [("Me", "tone440"), ("Speaker 1", "tone880"), ("Speaker 2", "tone1200")]
    assert s.state.mic_offset > 0 and s.state.sys_offset > 0
    raw = Path(s.state.raw_path).read_text()
    assert "Me: tone440" in raw and "Speaker 2: tone1200" in raw and "Статус: идёт запись" in raw
    assert Path(s.state.note_path).exists()
    assert p.step() is False   # nothing new


def test_echo_dropped_and_asr_error_marked(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    both = np.concatenate([silence(0.5), tone(2.0, 440), silence(3.0)])
    write(d, "mic.pcm", both)
    write(d, "sys.pcm", np.concatenate([silence(0.5), tone(2.0, 440), silence(1.0), tone(1.0, 660), silence(2.0)]))
    cfg, s, p = build(vault, tmp_path / "sessions", asr=FakeAsr(fail_on={660}))
    p.step()
    recs = sorted(s.state.records, key=lambda r: r.start)
    assert [r.channel for r in recs] == ["sys", "sys"]          # mic echo dropped
    assert recs[1].text is None and len(s.state.errors) == 1
    assert "[не распознано" in Path(s.state.raw_path).read_text()


def test_pending_mic_waits_for_sys_then_kept(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", np.concatenate([tone(1.0, 440), silence(1.0)]))
    write(d, "sys.pcm", silence(1.5))                # sys not yet past mic.end + 1 s
    cfg, s, p = build(vault, tmp_path / "sessions")
    p.step()
    assert s.state.records == [] and len(s.state.pending_mic) == 1
    with open(d / "sys.pcm", "ab") as f:
        f.write(to_pcm_bytes(silence(2.0)))
    p.step()
    assert [r.speaker for r in s.state.records] == ["Me"] and s.state.pending_mic == []


def test_resume_after_crash_is_idempotent(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(6.0))
    write(d, "sys.pcm", np.concatenate([tone(1.0, 880), silence(1.0), tone(1.0, 1200), silence(3.0)]))
    cfg, s, p = build(vault, tmp_path / "sessions")
    p.step()
    n = len(s.state.records)
    cfg2, s2, p2 = build(vault, tmp_path / "sessions")       # fresh pipeline from state.json
    assert p2.step() is False
    assert len(s2.state.records) == n
    assert p2.clusterer.n_speakers == 2


def test_finalize_reclusters_and_cleans_up(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(8.0))
    write(d, "sys.pcm", np.concatenate([tone(1.5, 880), silence(1.0), tone(1.5, 1200), silence(1.0), tone(1.5, 880), silence(1.5)]))
    cfg, s, p = build(vault, tmp_path / "sessions")
    p.step()
    (d / "STOP").touch()
    p.finalize()
    assert s.state.status == "finalized"
    assert [r.speaker for r in sorted(s.state.records, key=lambda r: r.start)] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    raw = Path(s.state.raw_path).read_text()
    assert "Статус: финализирован" in raw and "Говорящих на канале собеседников: 2" in raw
    assert not (d / "mic.pcm").exists() and not (d / "sys.pcm").exists()
    assert (d / "state.json").exists()


def test_finalize_keeps_audio_when_configured(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(1.0))
    write(d, "sys.pcm", silence(1.0))
    cfg, s, p = build(vault, tmp_path / "sessions", keep_audio=True)
    p.finalize()
    assert (d / "mic.pcm").exists()
