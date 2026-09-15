import json
import math

import numpy as np
import pytest
from transcriber.voices import VoiceLibrary


def v(*xs) -> list[float]:
    return [float(x) for x in xs]


def unit(*xs) -> list[float]:
    a = np.asarray(xs, dtype=np.float64)
    return [float(x) for x in a / np.linalg.norm(a)]


def lib(tmp_path, similarity: float = 0.45) -> VoiceLibrary:
    return VoiceLibrary(tmp_path / "voices", similarity=similarity)


def test_add_writes_a_file_per_person_and_matches_it_back(tmp_path):
    L = lib(tmp_path)
    L.add("Ярослав Серов", [v(1, 0, 0), v(0.9, 0.1, 0)], source="2026-09-15-1622:Speaker 2")
    p = tmp_path / "voices" / "Ярослав Серов.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["name"] == "Ярослав Серов" and data["count"] == 2
    assert data["sources"] == ["2026-09-15-1622:Speaker 2"]
    assert len(data["samples"]) == 2
    assert math.isclose(float(np.linalg.norm(data["centroid"])), 1.0, rel_tol=1e-6)

    fresh = lib(tmp_path)            # свежая библиотека читает то же самое с диска
    assert fresh.names() == ["Ярослав Серов"]
    hit = fresh.match(v(1, 0, 0))
    assert hit is not None and hit[0] == "Ярослав Серов" and hit[1] > 0.99


def test_centroid_is_the_mean_of_unit_vectors(tmp_path):
    L = lib(tmp_path)
    prof = L.add("Дарья", [v(2, 0, 0), v(0, 5, 0)], source="s:Speaker 1")
    assert prof.centroid == pytest.approx(unit(1, 1, 0), abs=1e-6)


def test_add_appends_to_an_existing_profile(tmp_path):
    L = lib(tmp_path)
    L.add("Дарья", [v(1, 0, 0)], source="a:Speaker 1")
    prof = lib(tmp_path).add("Дарья", [v(0, 1, 0), v(0, 1, 0)], source="b:Speaker 2")
    assert prof.count == 3 and len(prof.samples) == 3
    assert prof.sources == ["a:Speaker 1", "b:Speaker 2"]
    assert prof.centroid == pytest.approx(unit(1, 2, 0), abs=1e-6)


def test_match_returns_none_below_the_threshold(tmp_path):
    L = lib(tmp_path, similarity=0.45)
    L.add("Ярослав Серов", [v(1, 0, 0)], source="s:Speaker 1")
    assert L.match(v(0.4, 0.9, 0)) is None          # cos ≈ 0.41 < 0.45
    assert L.match(v(0.6, 0.8, 0)) is not None      # cos = 0.60


def test_match_requires_a_margin_over_the_runner_up(tmp_path):
    L = lib(tmp_path, similarity=0.45)
    L.add("Первый", [v(1, 0, 0)], source="s:Speaker 1")
    L.add("Второй", [v(0.99, 0.141, 0)], source="s:Speaker 2")   # почти тот же голос
    assert L.match(v(1, 0, 0)) is None                            # два кандидата в 0.05 друг от друга
    L.add("Третий", [v(0, 0, 1)], source="s:Speaker 3")
    hit = L.match(v(0, 0, 1))                                     # а этот отрывается с запасом
    assert hit is not None and hit[0] == "Третий"


def test_match_on_an_empty_library_is_none(tmp_path):
    assert lib(tmp_path).match(v(1, 0, 0)) is None
    assert lib(tmp_path).names() == []


def test_samples_are_capped_at_50_most_recent_but_count_keeps_growing(tmp_path):
    L = lib(tmp_path)
    L.add("Дарья", [v(1, 0, 0)] * 40, source="a:Speaker 1")
    prof = L.add("Дарья", [v(0, 1, 0)] * 20, source="b:Speaker 1")
    assert prof.count == 60 and len(prof.samples) == 50
    assert prof.samples[-1] == pytest.approx(v(0, 1, 0))
    assert prof.samples[:30] == [pytest.approx(v(1, 0, 0))] * 30    # свежие вытеснили самые старые
    assert json.loads((tmp_path / "voices" / "Дарья.json").read_text(encoding="utf-8"))["count"] == 60


def test_name_is_sanitised_for_the_filename_but_kept_intact(tmp_path):
    L = lib(tmp_path)
    prof = L.add("Иван / Петров", [v(1, 0, 0)], source="s:Speaker 1")
    assert prof.name == "Иван / Петров"
    assert [p.name for p in (tmp_path / "voices").iterdir()] == ["Иван Петров.json"]
    assert lib(tmp_path).names() == ["Иван / Петров"]      # имя читается из тела файла, не из пути


def test_profiles_of_another_dimension_are_ignored_not_crashed(tmp_path):
    L = lib(tmp_path)
    L.add("Восьмимерный", [v(*([1] + [0] * 7))], source="s:Speaker 1")
    L.add("Трёхмерный", [v(1, 0, 0)], source="s:Speaker 2")
    hit = L.match(v(1, 0, 0))
    assert hit is not None and hit[0] == "Трёхмерный"


def test_junk_vectors_never_match(tmp_path):
    L = lib(tmp_path)
    L.add("Дарья", [v(1, 0, 0)], source="s:Speaker 1")
    assert L.match(v(0, 0, 0)) is None
    assert L.match(v(float("nan"), 0, 0)) is None
    assert L.add("Пустой", [], source="s:Speaker 9") is None      # учить не на чем


def test_a_broken_json_file_does_not_break_the_library(tmp_path):
    L = lib(tmp_path)
    L.add("Дарья", [v(1, 0, 0)], source="s:Speaker 1")
    (tmp_path / "voices" / "мусор.json").write_text("{ это не json", encoding="utf-8")
    fresh = lib(tmp_path)
    assert fresh.names() == ["Дарья"] and fresh.match(v(1, 0, 0)) is not None
