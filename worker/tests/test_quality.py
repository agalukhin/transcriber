from transcriber.asr import build_prompt
from transcriber.quality import (
    filler_only,
    foreign,
    junk_reason,
    prompt_leak,
    prompt_vocabulary,
    repetitive,
    tokenize,
)

GLOSSARY = ["Владислав", "Магнит", "Омни", "CDP", "Feature store", "АБ-тест", "Толк",
            "Егор Спирин", "Ярослав Серов", "Катя Гладникова", "Саша Галухин"]
VOCAB = prompt_vocabulary([], GLOSSARY)


def test_tokenize_lowercases_and_keeps_digits():
    assert tokenize("Привет, CDP-2 — это Магнит!") == ["привет", "cdp", "2", "это", "магнит"]


def test_vocabulary_covers_prompt_keywords_participants_and_glossary():
    assert {"участники", "термины", "владислав", "магнит", "галухин", "cdp"} <= VOCAB
    assert "спирин" in prompt_vocabulary(["Егор Спирин"], [])
    # словарь строится из исходных списков, а не из обрезанного по 600 символам промпта
    long_vocab = prompt_vocabulary([], [f"термин{i}" for i in range(200)])
    assert "термин199" in long_vocab and len(build_prompt([], [f"термин{i}" for i in range(200)])) <= 600


def test_prompt_leak_catches_prompt_echoes_from_the_real_meeting():
    for text in ["Термины, Саша Галухин.",
                 "Владислав, Магнит, Омни, CDP, Факт, Катя Галухин.",
                 "Участники Галухин, Магнит, Офлайн, Магнит, Офлайн, Магнит, Офлайн, Магнит.",
                 "Владислав Свердлов, Катя Галухин. Владислав Свердлов, Катя Галухин."]:
        assert prompt_leak(text, VOCAB), text


def test_prompt_leak_spares_normal_speech_and_short_phrases():
    assert not prompt_leak("Давайте тогда это запланируем, начнём с 1 октября", VOCAB)
    assert not prompt_leak("В Магните мы считаем CDP главным приоритетом на квартал", VOCAB)
    assert not prompt_leak("Магнит Омни", VOCAB)          # меньше трёх слов — не трогаем
    assert not prompt_leak("", VOCAB)


def test_repetitive_catches_consecutive_ngrams_and_low_variety():
    assert repetitive("Магнит, Офлайн, Магнит, Офлайн, Магнит, Офлайн")
    assert repetitive("да да да")
    assert repetitive("так так так так вот так так так")      # уникальных слов < 35 %
    assert not repetitive("Давайте начнём с первого октября, тогда всё сойдётся")
    assert not repetitive("да да, поехали")


def test_foreign_drops_english_on_a_russian_call_but_keeps_latin_acronyms():
    assert foreign("News Sl coming up.")
    assert not foreign("Все, продолжаем. Sorry.")
    assert not foreign("Мы отдадим SQL из CDP")
    assert not foreign("ok fine", language="ru")             # меньше трёх слов
    assert not foreign("News Sl coming up.", language="en")


def test_filler_only_drops_pure_hesitation():
    assert filler_only("Аааа... Ааа...")
    assert filler_only("Ну")
    assert filler_only("Э-э, ммм")
    assert not filler_only("Ну да, конечно")
    assert not filler_only("")


def test_junk_reason_names_the_rule_and_passes_clean_text():
    assert junk_reason("Термины, Саша Галухин.", VOCAB) == "prompt leak"
    assert junk_reason("Аааа... Ааа...", VOCAB) == "filler"
    assert junk_reason("News Sl coming up.", VOCAB) == "foreign"
    assert junk_reason("Хорошо, давай обсудим контракт", VOCAB) is None
    assert junk_reason("", VOCAB) is None
