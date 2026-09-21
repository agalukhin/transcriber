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


MIC_LONG = ("Когда будем отдавать вам сегмент, отдадим SQL, которым его собирали мы. То есть он будет похож "
            "на ваш, но может быть будет немножко другой. То есть там будут, возможно, чуть-чуть другие "
            "джойны, чуть-чуть другие фильтры. Друзья, я прошу прощения, буквально полминутки мне нужно "
            "для теста. Вы можете своим голосом представиться?")
SYS_LONG = ("Когда будем отдавать вам сегмент, отдадим SQL, которым его собирали мы. То есть он будет похож "
            "на ваш, но может быть будет немножко другой. То есть вам будут, возможно, чуть другие джойны, "
            "чуть другие фильтры.")


def test_echo_by_overlap_when_mic_is_barely_longer():
    """Реальный случай: mic 40.7–52.0, sys 40.8–51.5, схожесть 0.72 — ловит первое правило."""
    m = mic(40.7, 52.0, MIC_LONG)
    others = [sys_(40.8, 51.5, SYS_LONG)]
    assert 0.6 <= similarity(m.text, others[0].text) < 0.8
    assert overlap_fraction(m, others) > 0.9
    assert is_echo(m, others, 0.8, 0.6, strong=0.8)


def test_echo_by_strong_similarity_despite_short_overlap():
    """Реальный случай: микрофон сегментирован длиннее системного, доля перекрытия падает ниже 0.8."""
    m = mic(62.7, 65.9, "Хорошо, меня зовут Ярослав. Я Владислав.")
    others = [sys_(62.7, 64.6, "Хорошо. Меня зовут Ярослав.")]
    assert overlap_fraction(m, others) < 0.8            # первое правило не срабатывает
    assert is_echo(m, others, 0.8, 0.6, strong=0.8)
    assert not is_echo(m, others, 0.8, 0.6, strong=0.9)


def test_echo_by_strong_similarity_on_identical_short_line():
    """mic 60–70 и sys 62–64 с тем же текстом: перекрытие 0.2, схожесть 1.0 — второе правило."""
    m = mic(60.0, 70.0, "Хорошо.")
    others = [sys_(62.0, 64.0, "Хорошо.")]
    assert overlap_fraction(m, others) == 0.2
    assert is_echo(m, others, 0.8, 0.6, strong=0.8)


def test_strong_rule_still_needs_actual_time_overlap():
    m = mic(60.0, 70.0, "Хорошо, меня зовут Ярослав")
    assert not is_echo(m, [sys_(80.0, 84.0, "Хорошо, меня зовут Ярослав.")], 0.8, 0.6, strong=0.8)


def test_stock_phrase_on_mic_while_speakers_talk_is_residue():
    from transcriber.echo import is_residue
    others = [sys_(100.0, 130.0, "Дальше у нас была история с сегментами и фильтрами.")]
    assert is_residue(mic(110.0, 110.6, "Спасибо."), others)
    assert is_residue(mic(112.0, 112.9, "Продолжение следует..."), others)
    assert is_residue(mic(113.0, 113.4, "Спасибо за просмотр!"), others)
    # владелец благодарит в тишине — это его реплика
    assert not is_residue(mic(200.0, 200.6, "Спасибо."), others)
    # содержательная фраза поверх чужой речи — не остаток
    assert not is_residue(mic(110.0, 112.0, "Спасибо, Паша, за обзор, у меня вопрос."), others)
    # длинная «реплика» из одного слова — не наш случай, пусть решают другие правила
    assert not is_residue(mic(110.0, 118.0, "Спасибо."), others)
    assert not is_residue(mic(110.0, 110.6, None), others)
