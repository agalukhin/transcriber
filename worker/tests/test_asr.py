from transcriber.asr import Segment, build_prompt, clean_text


def test_clean_strips_hallucinations_and_space():
    h = ["Продолжение следует", "Субтитры сделал DimaTorzok"]
    assert clean_text("  Мы посчитаем.  Продолжение следует... ", h) == "Мы посчитаем."
    assert clean_text("Субтитры сделал DimaTorzok", h) == ""
    assert clean_text("субтитры сделал dimatorzok", h) == ""
    assert clean_text("Обычный текст", h) == "Обычный текст"


def test_prompt_is_terms_only_by_default():
    """Имена — главный источник утечки промпта, поэтому по умолчанию их в промпте нет."""
    assert build_prompt(["Егор Спирин"], ["CDP", "Омни"]) == "Термины: CDP, Омни."
    assert build_prompt(["Егор Спирин"], []) == ""


def test_prompt_adds_participants_only_when_asked():
    p = build_prompt(["Егор Спирин"], ["CDP"], include_participants=True)
    assert p == "Участники: Владислав, Егор Спирин. Термины: CDP."
    assert build_prompt([], ["CDP"], include_participants=True) == "Термины: CDP."


def test_prompt_caps_glossary_at_twelve_terms():
    p = build_prompt([], [f"термин{i}" for i in range(200)])
    assert p.startswith("Термины: термин0,") and "термин11" in p
    assert "термин12" not in p and len(p) <= 600


def test_prompt_caps_even_without_comma_in_truncated_prefix():
    long_term = "x" * 700
    p = build_prompt([], [long_term])
    assert len(p) <= 600


def test_clean_leaves_no_punctuation_debris():
    h = ["Продолжение следует"]
    assert clean_text("Ок, продолжение следует, дальше", h) == "Ок, дальше"


def _transcriber(monkeypatch, result):
    """Transcriber с подменённым mlx_whisper — настоящую модель не грузим."""
    import sys

    from transcriber.asr import Transcriber

    class FakeWhisper:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, audio, **kw):
            self.kwargs = kw
            return result

    fake = FakeWhisper()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    return Transcriber("model", "ru", ["Продолжение следует"]), fake


def _seg(text, start=0.0, end=1.0, no_speech=0.0, logprob=-0.3):
    return {"text": text, "start": start, "end": end, "no_speech_prob": no_speech, "avg_logprob": logprob}


def test_transcriber_returns_a_segment_per_whisper_segment(monkeypatch):
    import numpy as np

    result = {"text": " Первое. Второе.", "segments": [
        _seg(" Первое.", 0.0, 1.5), _seg(" Второе.", 2.0, 3.5)]}
    t, fake = _transcriber(monkeypatch, result)
    got = t(np.zeros(16000, dtype="float32"), "промпт")
    assert got == [Segment(0.0, 1.5, "Первое."), Segment(2.0, 3.5, "Второе.")]
    assert fake.kwargs["initial_prompt"] == "промпт"
    assert fake.kwargs["condition_on_previous_text"] is True   # блок длинный, контекст внутри него нужен


def test_transcriber_drops_silent_and_low_confidence_segments(monkeypatch):
    import numpy as np

    result = {"text": "", "segments": [
        _seg(" Первое.", 0.0, 1.0),
        _seg(" Тишина.", 1.0, 2.0, no_speech=0.9),
        _seg(" Бред.", 2.0, 3.0, logprob=-1.5),
        _seg(" Второе.", 3.0, 4.0),
    ]}
    t, _ = _transcriber(monkeypatch, result)
    assert [s.text for s in t(np.zeros(16000, dtype="float32"))] == ["Первое.", "Второе."]


def test_transcriber_cleans_hallucinations_and_drops_empty_segments(monkeypatch):
    import numpy as np

    result = {"text": "", "segments": [
        _seg(" Продолжение следует", 0.0, 1.0), _seg(" Есть текст.", 1.0, 2.0)]}
    t, _ = _transcriber(monkeypatch, result)
    assert [s.text for s in t(np.zeros(16000, dtype="float32"))] == ["Есть текст."]


def test_transcriber_falls_back_to_one_segment_over_the_whole_block(monkeypatch):
    import numpy as np

    t, _ = _transcriber(monkeypatch, {"text": " Есть текст. Продолжение следует"})
    assert t(np.zeros(32000, dtype="float32")) == [Segment(0.0, 2.0, "Есть текст.")]


def test_transcriber_survives_segments_without_time_and_quality_fields(monkeypatch):
    import numpy as np

    t, _ = _transcriber(monkeypatch, {"text": " Текст.", "segments": [{"text": " Текст."}]})
    assert t(np.zeros(16000, dtype="float32")) == [Segment(0.0, 0.0, "Текст.")]
