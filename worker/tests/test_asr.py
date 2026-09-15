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
