import json
from pathlib import Path

import numpy as np
from transcriber.asr import Segment
from transcriber.config import Config
from transcriber.pipeline import Pipeline
from transcriber.session import Session
from tests.conftest import (FakeAsr, FakeEmbedder, SR, dominant_freq, fake_prob, make_session, silence,
                            to_pcm_bytes, tone)


# блок на реплику: тесты про эхо, говорящих и фильтры, а не про размер блока
FAST = dict(block_min_s=0.5, block_gap_s=0.5, block_idle_s=0.5)


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
    sys_ = np.concatenate([silence(2.0), tone(1.5, 880), silence(1.0), tone(1.5, 1200), silence(4.5)])
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
    both = np.concatenate([silence(0.5), tone(2.0, 440), silence(1.0)])
    write(d, "mic.pcm", both)
    write(d, "sys.pcm", np.concatenate([silence(0.5), tone(2.0, 440), silence(1.0), tone(1.0, 660), silence(1.0)]))
    cfg, s, p = build(vault, tmp_path / "sessions", asr=FakeAsr(fail_on={660}), **FAST)
    p.step()
    recs = sorted(s.state.records, key=lambda r: r.start)
    assert [r.channel for r in recs] == ["sys", "sys"]          # mic echo dropped
    assert recs[1].text is None and len(s.state.errors) == 1
    assert "[не распознано" in Path(s.state.raw_path).read_text()


def test_pending_mic_waits_for_sys_then_kept(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", np.concatenate([tone(1.0, 440), silence(1.0)]))
    write(d, "sys.pcm", silence(1.5))                # sys not yet past mic.end + 1 s
    cfg, s, p = build(vault, tmp_path / "sessions", **FAST)
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
    cfg, s, p = build(vault, tmp_path / "sessions", **FAST)
    p.step()
    n = len(s.state.records)
    cfg2, s2, p2 = build(vault, tmp_path / "sessions", **FAST)   # fresh pipeline from state.json
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


def test_finalize_flushes_open_sys_utterance(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(3.0))
    write(d, "sys.pcm", np.concatenate([silence(0.5), tone(2.5, 880)]))   # запись обрывается на речи
    cfg, s, p = build(vault, tmp_path / "sessions", keep_audio=True)
    p.step()
    assert s.state.records == []            # реплика ещё открыта
    p.finalize()
    recs = [r for r in s.state.records if r.channel == "sys"]
    assert len(recs) == 1 and recs[0].text == "tone880"
    assert "tone880" in Path(s.state.raw_path).read_text()
    cfg2, s2, p2 = build(vault, tmp_path / "sessions", keep_audio=True)   # повтор не дублирует
    p2.finalize()
    assert len([r for r in s2.state.records if r.channel == "sys"]) == 1


def test_pending_mic_flushed_without_sys_track(vault, tmp_path):
    """sys.pcm может вовсе не появиться — реплики владельца не должны висеть до финализации."""
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", np.concatenate([silence(0.5), tone(1.0, 440), silence(5.0)]))
    cfg, s, p = build(vault, tmp_path / "sessions")
    p.step()
    assert [(r.channel, r.text) for r in s.state.records] == [("mic", "tone440")]
    assert s.state.pending_mic == []


def test_step_reads_bounded_chunks_and_reports_backlog(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(1.0))
    write(d, "sys.pcm", silence(90.0))
    cfg, s, p = build(vault, tmp_path / "sessions")
    p.step()
    assert s.state.sys_offset <= 30 * SR + 512
    assert p.has_backlog() is True
    for _ in range(3):
        p.step()
    assert p.has_backlog() is False


def test_state_saved_only_when_something_moved(vault, tmp_path, monkeypatch):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(2.0))
    write(d, "sys.pcm", silence(2.0))
    cfg, s, p = build(vault, tmp_path / "sessions")
    p.step()
    calls: list[int] = []
    monkeypatch.setattr(s, "save_state", lambda: calls.append(1))
    assert p.step() is False
    assert calls == []


def test_short_sys_utterance_inherits_first_embedded_label(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(9.0))
    write(d, "sys.pcm", np.concatenate([
        silence(0.2), tone(0.4, 880), silence(1.0),     # короткая — эмбеддинга не будет
        tone(1.5, 1200), silence(1.0),                  # первая с эмбеддингом
        tone(1.5, 880), silence(1.5),
    ]))
    cfg, s, p = build(vault, tmp_path / "sessions")
    p.step()
    (d / "STOP").touch()
    p.finalize()
    recs = sorted([r for r in s.state.records if r.channel == "sys"], key=lambda r: r.start)
    assert len(recs) == 3 and recs[0].emb is None
    first_emb = next(r for r in recs if r.emb is not None)
    assert recs[0].speaker == first_emb.speaker
    assert recs[2].speaker != first_emb.speaker


def test_restart_inside_utterance_produces_one_record(vault, tmp_path):
    sessions = tmp_path / "sessions"
    whole = np.concatenate([silence(0.5), tone(2.0, 880), silence(1.5)])
    ref_dir = make_session(sessions, session_id="ref", title="Эталон")
    write(ref_dir, "mic.pcm", silence(4.0))
    write(ref_dir, "sys.pcm", whole)
    _, ref_s, ref_p = build(vault, sessions, session_id="ref", **FAST)
    ref_p.step()

    d = make_session(sessions, session_id="split", title="С обрывом")
    write(d, "mic.pcm", silence(4.0))
    write(d, "sys.pcm", whole[: int(1.5 * SR)])          # рестарт посреди реплики
    _, s, p = build(vault, sessions, session_id="split", **FAST)
    p.step()
    with open(d / "sys.pcm", "ab") as f:
        f.write(to_pcm_bytes(whole[int(1.5 * SR):]))
    _, s2, p2 = build(vault, sessions, session_id="split", **FAST)   # свежий Pipeline из state.json
    p2.step()

    recs = [r for r in s2.state.records if r.channel == "sys"]
    ref = [r for r in ref_s.state.records if r.channel == "sys"]
    assert len(ref) == 1 and len(recs) == 1
    assert recs[0].text == ref[0].text
    assert abs(recs[0].start - ref[0].start) <= 512 / SR + 1e-6


def test_pending_mic_survives_restart(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", np.concatenate([tone(1.0, 440), silence(1.0)]))
    write(d, "sys.pcm", silence(1.5))
    cfg, s, p = build(vault, tmp_path / "sessions", **FAST)
    p.step()
    assert len(s.state.pending_mic) == 1
    assert len(json.loads((d / "state.json").read_text())["pending_mic"]) == 1
    with open(d / "sys.pcm", "ab") as f:
        f.write(to_pcm_bytes(silence(2.0)))
    cfg2, s2, p2 = build(vault, tmp_path / "sessions", **FAST)   # свежий Pipeline
    assert len(s2.state.pending_mic) == 1
    p2.step()
    assert [(r.channel, r.text) for r in s2.state.records] == [("mic", "tone440")]
    assert s2.state.pending_mic == []


def test_hallucinated_prompt_and_filler_are_dropped(vault, tmp_path, caplog):
    """Whisper пересказывает промпт и мычит на неразборчивом — в транскрипт это не попадает."""
    d = make_session(tmp_path / "sessions", participants=["Саша Галухин"])
    write(d, "mic.pcm", silence(10.0))
    write(d, "sys.pcm", np.concatenate([
        tone(1.5, 880), silence(1.0), tone(1.5, 1200), silence(1.0), tone(1.5, 660), silence(4.5),
    ]))
    texts = {880: "Термины, Саша Галухин.", 1200: "Аааа... Ааа...", 660: "Давайте вернёмся к контракту"}
    asr = FakeAsr(texts=texts)
    cfg, s, p = build(vault, tmp_path / "sessions", asr=asr)
    with caplog.at_level("INFO", logger="transcriber"):
        p.step()
    recs = sorted([r for r in s.state.records if r.channel == "sys"], key=lambda r: r.start)
    assert [r.text for r in recs] == ["", "", "Давайте вернёмся к контракту"]
    assert "prompt leak" in caplog.text and "filler" in caplog.text
    assert len(asr.prompts) == 1                      # три реплики ушли в модель одним блоком
    body = Path(s.state.raw_path).read_text().split("## Полный транскрипт", 1)[1]
    assert "Галухин" not in body and "Аааа" not in body
    assert body.strip().endswith("Давайте вернёмся к контракту")


def test_asr_failure_is_still_marked_unrecognised_not_dropped(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(5.0))
    write(d, "sys.pcm", np.concatenate([tone(1.5, 880), silence(2.0)]))
    cfg, s, p = build(vault, tmp_path / "sessions", asr=FakeAsr(fail_on={880}), **FAST)
    p.step()
    assert [r.text for r in s.state.records] == [None]
    assert "[не распознано" in Path(s.state.raw_path).read_text()


class NearEmbedder:
    """660 Гц — похожий, но не совпадающий голос: отдельный кластер, который надо подклеить."""

    VECTORS = {880: (1.0, 0.0, 0.0), 1200: (0.0, 1.0, 0.0), 660: (0.25, 0.0, 0.97)}

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return np.array(self.VECTORS[dominant_freq(audio)], dtype=np.float32)


def test_finalize_absorbs_single_utterance_speaker(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(13.0))
    write(d, "sys.pcm", np.concatenate([
        tone(1.5, 880), silence(1.0), tone(1.5, 1200), silence(1.0),
        tone(1.5, 880), silence(1.0), tone(1.5, 1200), silence(1.0),
        tone(1.5, 660), silence(1.5),
    ]))
    cfg = Config(vault_root=vault, sessions_dir=tmp_path / "sessions")
    s = Session.load(tmp_path / "sessions" / "2026-09-15-1430")
    p = Pipeline(cfg, s, asr=FakeAsr(), embed=NearEmbedder(), prob_factory=lambda: fake_prob)
    p.step()
    (d / "STOP").touch()
    p.finalize()
    recs = sorted([r for r in s.state.records if r.channel == "sys"], key=lambda r: r.start)
    assert [r.speaker for r in recs] == ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2", "Speaker 1"]
    assert len(set(s.state.speaker_of_emb)) == 2
    assert "Говорящих на канале собеседников: 2" in Path(s.state.raw_path).read_text()


class LeakyAsr:
    """Whisper с промптом пересказывает промпт, без промпта слышит живую речь."""

    def __init__(self, always_leaks: bool = False):
        self.always_leaks = always_leaks
        self.prompts: list[str] = []

    def __call__(self, audio: np.ndarray, prompt: str = "") -> list[Segment]:
        self.prompts.append(prompt)
        leak = "Термины, Саша Галухин, Магнит, Омни."
        text = leak if (prompt or self.always_leaks) else "Давайте вернёмся к контракту"
        return [Segment(0.0, len(audio) / SR, text)]


def test_prompt_leak_is_retried_without_prompt(vault, tmp_path, caplog):
    """Утечка промпта — не повод терять реплику: переспрашиваем модель без промпта."""
    d = make_session(tmp_path / "sessions", participants=["Саша Галухин"])
    write(d, "mic.pcm", silence(5.0))
    write(d, "sys.pcm", np.concatenate([tone(1.5, 880), silence(2.0)]))
    asr = LeakyAsr()
    cfg, s, p = build(vault, tmp_path / "sessions", asr=asr, glossary=["Магнит", "Омни", "CDP"], **FAST)
    with caplog.at_level("INFO", logger="transcriber"):
        p.step()
    assert [r.text for r in s.state.records if r.channel == "sys"] == ["Давайте вернёмся к контракту"]
    assert asr.prompts[0] and asr.prompts[1] == ""
    assert "retry without prompt" in caplog.text
    assert "Давайте вернёмся к контракту" in Path(s.state.raw_path).read_text()


def test_prompt_leak_on_retry_too_is_dropped(vault, tmp_path, caplog):
    d = make_session(tmp_path / "sessions", participants=["Саша Галухин"])
    write(d, "mic.pcm", silence(5.0))
    write(d, "sys.pcm", np.concatenate([tone(1.5, 880), silence(2.0)]))
    asr = LeakyAsr(always_leaks=True)
    cfg, s, p = build(vault, tmp_path / "sessions", asr=asr, glossary=["Магнит", "Омни", "CDP"], **FAST)
    with caplog.at_level("INFO", logger="transcriber"):
        p.step()
    assert [r.text for r in s.state.records if r.channel == "sys"] == [""]
    assert asr.prompts == [asr.prompts[0], ""]
    assert "retry without prompt" in caplog.text and "prompt leak" in caplog.text
    assert "Галухин" not in Path(s.state.raw_path).read_text().split("## Полный транскрипт", 1)[1]


def test_two_utterances_two_seconds_apart_become_one_block(vault, tmp_path):
    """Пауза в 2 с блок не режет: минимум в 45 с не набран — обе реплики уходят в модель разом."""
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(10.0))
    write(d, "sys.pcm", np.concatenate([silence(0.5), tone(1.5, 880), silence(2.0), tone(1.5, 1200), silence(4.5)]))
    asr = FakeAsr()
    cfg, s, p = build(vault, tmp_path / "sessions", asr=asr)
    assert p.step() is True
    assert len(asr.prompts) == 1                      # один вызов модели на две реплики
    recs = sorted([r for r in s.state.records if r.channel == "sys"], key=lambda r: r.start)
    assert [(r.text, r.speaker) for r in recs] == [("tone880", "Speaker 1"), ("tone1200", "Speaker 2")]
    assert abs(recs[0].start - 0.5) < 0.06 and abs(recs[0].end - 2.0) < 0.06   # времена вернулись в дорожку
    assert abs(recs[1].start - 4.0) < 0.06 and abs(recs[1].end - 5.5) < 0.06


def test_resume_inside_open_block_retranscribes_it_without_duplicates(vault, tmp_path):
    """Падение посреди блока: откатываемся к его началу, переслушиваем один блок, дублей нет."""
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(8.0))
    write(d, "sys.pcm", np.concatenate([silence(0.5), tone(1.5, 880), silence(2.0), tone(1.5, 1200), silence(1.0)]))
    cfg, s, p = build(vault, tmp_path / "sessions", keep_audio=True)
    p.step()
    assert s.state.records == []                                  # блок ещё открыт, текста нет
    assert abs(s.state.sys_offset / SR - 0.4) < 0.06              # откат к началу первой реплики блока
    cfg2, s2, p2 = build(vault, tmp_path / "sessions", keep_audio=True)   # свежий Pipeline из state.json
    p2.finalize()
    recs = sorted([r for r in s2.state.records if r.channel == "sys"], key=lambda r: r.start)
    assert [r.text for r in recs] == ["tone880", "tone1200"]


def test_previous_block_text_is_carried_into_the_next_prompt(vault, tmp_path):
    d = make_session(tmp_path / "sessions")
    write(d, "mic.pcm", silence(6.0))
    write(d, "sys.pcm", np.concatenate([tone(1.0, 880), silence(1.0), tone(1.0, 1200), silence(1.0)]))
    asr = FakeAsr()
    cfg, s, p = build(vault, tmp_path / "sessions", asr=asr, **FAST)
    p.step()
    base = p.prompt
    assert asr.prompts == [base, f"{base} … tone880"]  # хвост прошлого блока — штатный способ whisper

    asr2 = FakeAsr()
    d2 = make_session(tmp_path / "sessions", session_id="2026-09-15-1500", title="Без контекста")
    write(d2, "mic.pcm", silence(6.0))
    write(d2, "sys.pcm", np.concatenate([tone(1.0, 880), silence(1.0), tone(1.0, 1200), silence(1.0)]))
    _, _, p2 = build(vault, tmp_path / "sessions", session_id="2026-09-15-1500", asr=asr2,
                     carry_context=False, **FAST)
    p2.step()
    assert asr2.prompts == [base, base]
