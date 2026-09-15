from transcriber.echo import is_echo, overlap_fraction, similarity
from transcriber.session import Rec


def mic(s, e, t):
    return Rec("mic", s, e, t, "Me", None)


def sys_(s, e, t):
    return Rec("sys", s, e, t, "Speaker 1", None)


def test_similarity_ignores_case_and_punct():
    assert similarity("Привет, слышно?", "привет слышно") > 0.9
    assert similarity("Привет", "Совсем другое") < 0.4


def test_overlap_fraction():
    assert overlap_fraction(mic(0, 2, ""), [sys_(1, 3, "")]) == 0.5
    assert overlap_fraction(mic(0, 2, ""), [sys_(0, 1, ""), sys_(1, 2, "")]) == 1.0
    assert overlap_fraction(mic(0, 2, ""), [sys_(5, 6, "")]) == 0.0


def test_echo_detected_when_overlapping_and_similar():
    m = mic(10.0, 13.0, "Мы посчитаем параметры на следующей неделе")
    others = [sys_(9.8, 13.2, "Мы посчитаем параметры на следующей неделе.")]
    assert is_echo(m, others, overlap=0.8, sim=0.6)


def test_not_echo_when_text_differs_or_no_overlap():
    m = mic(10.0, 13.0, "Да, согласен")
    assert not is_echo(m, [sys_(9.8, 13.2, "Мы посчитаем параметры на следующей неделе")], 0.8, 0.6)
    assert not is_echo(m, [sys_(20, 23, "Да, согласен")], 0.8, 0.6)
    assert not is_echo(mic(0, 1, None), [sys_(0, 1, "x")], 0.8, 0.6)
