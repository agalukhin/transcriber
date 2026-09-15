from transcriber.asr import build_prompt, clean_text


def test_clean_strips_hallucinations_and_space():
    h = ["Продолжение следует", "Субтитры сделал DimaTorzok"]
    assert clean_text("  Мы посчитаем.  Продолжение следует... ", h) == "Мы посчитаем."
    assert clean_text("Субтитры сделал DimaTorzok", h) == ""
    assert clean_text("субтитры сделал dimatorzok", h) == ""
    assert clean_text("Обычный текст", h) == "Обычный текст"


def test_prompt_has_participants_and_glossary_capped():
    p = build_prompt(["Егор Спирин"], ["CDP", "Омни"])
    assert p == "Участники: Владислав, Егор Спирин. Термины: CDP, Омни."
    long = build_prompt([], [f"термин{i}" for i in range(200)])
    assert len(long) <= 600


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


def _seg(text, no_speech=0.0, logprob=-0.3):
    return {"text": text, "no_speech_prob": no_speech, "avg_logprob": logprob}


def test_transcriber_keeps_result_text_when_all_segments_are_solid(monkeypatch):
    import numpy as np

    result = {"text": " Первое. Второе.", "segments": [_seg(" Первое."), _seg(" Второе.")]}
    t, fake = _transcriber(monkeypatch, result)
    assert t(np.zeros(16000, dtype="float32"), "промпт") == "Первое. Второе."
    assert fake.kwargs["initial_prompt"] == "промпт"


def test_transcriber_drops_silent_and_low_confidence_segments(monkeypatch):
    import numpy as np

    result = {"text": " Первое. Тишина. Бред. Второе.", "segments": [
        _seg(" Первое."),
        _seg(" Тишина.", no_speech=0.9),
        _seg(" Бред.", logprob=-1.5),
        _seg(" Второе."),
    ]}
    t, _ = _transcriber(monkeypatch, result)
    assert t(np.zeros(16000, dtype="float32")) == "Первое. Второе."


def test_transcriber_falls_back_to_text_without_segments(monkeypatch):
    import numpy as np

    t, _ = _transcriber(monkeypatch, {"text": " Есть текст. Продолжение следует"})
    assert t(np.zeros(16000, dtype="float32")) == "Есть текст."


def test_transcriber_survives_segments_without_quality_fields(monkeypatch):
    import numpy as np

    t, _ = _transcriber(monkeypatch, {"text": " Текст.", "segments": [{"text": " Текст."}]})
    assert t(np.zeros(16000, dtype="float32")) == "Текст."
