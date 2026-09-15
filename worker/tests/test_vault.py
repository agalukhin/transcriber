from datetime import datetime
from transcriber.config import Config
from transcriber.session import Meta, Rec, Session, State
from transcriber.vault import ensure_stub, raw_filename, note_filename, render_raw, slug, write_raw


def meta(title="Синк по CDP / параметры", app="kontur.talk"):
    return Meta(id="2026-09-15-1430", started_at=datetime.fromisoformat("2026-09-15T14:30:00+03:00"),
                app=app, title=title, participants=["Егор Спирин", "Ярослав Серов"])


def test_slug_and_filenames():
    assert slug("Синк по CDP / параметры") == "Синк-по-CDP-параметры"
    assert raw_filename(meta()) == "Local-2026-09-15-1430-Синк-по-CDP-параметры.md"
    assert note_filename(meta()) == "MTG-2026-09-15 1430 · Синк по CDP параметры.md"
    assert raw_filename(meta(title="")) == "Local-2026-09-15-1430-Встреча-в-Толке.md"
    assert note_filename(meta(title="", app="zoom.us")) == "MTG-2026-09-15 1430 · Встреча в Zoom.md"


def test_render_raw_body_and_markers():
    recs = [
        Rec("mic", 0.5, 2.0, "Привет.", "Me", None),
        Rec("sys", 2.1, 4.0, "Привет, слышно?", "Speaker 1", 0),
        Rec("sys", 4.5, 5.0, None, "Speaker 1", None),
        Rec("sys", 301.0, 303.0, "Дальше.", "Speaker 2", 1),
    ]
    text = render_raw(meta(), recs, n_speakers=2, status_text="идёт запись", marker_every_s=300)
    assert text.startswith("---\ntype: raw\nsource: local\nsource_date: 2026-09-15\nprojects: []\ntags: [raw]\n---\n")
    assert "# Синк по CDP / параметры — 2026-09-15 14:30" in text
    assert "Участники: Егор Спирин, Ярослав Серов" in text
    assert "Говорящих на канале собеседников: 2" in text and "Статус: идёт запись" in text
    body = text.split("## Полный транскрипт\n\n")[1]
    assert body.split("\n\n")[0:2] == ["Me: Привет.", "Speaker 1: Привет, слышно?"]
    assert "[не распознано 00:04–00:05]" in body
    assert "— 00:05 —\n\nSpeaker 2: Дальше." in body
    assert body.count("— 00:00 —") == 0


def test_write_raw_and_stub(vault):
    cfg = Config(vault_root=vault)
    s = Session(vault / "s", meta(), State())
    p = write_raw(cfg, s, status_text="идёт запись")
    assert p == vault / "Memory/50 · Raw/Local/Local-2026-09-15-1430-Синк-по-CDP-параметры.md"
    assert p.exists() and s.state.raw_path == str(p)
    n = ensure_stub(cfg, s)
    assert n == vault / "Memory/30 · Meetings/MTG-2026-09-15 1430 · Синк по CDP параметры.md"
    body = n.read_text()
    assert "id: MTG-2026-09-15-1430\n" in body
    assert "participants: [Владислав, Егор Спирин, Ярослав Серов]\n" in body
    assert 'transcript: "[[Local-2026-09-15-1430-Синк-по-CDP-параметры]]"\n' in body
    assert "tags: [meeting, stub]\n" in body
    assert "## Мысли Владислава\n" in body and "<!-- дистиллят: ожидает дейли-рутин -->" in body
    n.write_text(body + "\nмоя мысль\n")
    assert ensure_stub(cfg, s) == n and "моя мысль" in n.read_text()   # never overwritten


def test_title_of_falls_back_when_title_is_only_separators():
    assert raw_filename(meta(title="///")) == raw_filename(meta(title=""))


def test_filename_title_length_is_capped():
    long_title = "А" * 200
    assert len(raw_filename(meta(title=long_title))) < 150
    assert len(note_filename(meta(title=long_title))) < 150


def test_app_label_and_locative_share_one_source():
    from transcriber.vault import app_label, app_locative

    assert app_label("kontur.talk") == "Толк"
    assert app_locative("kontur.talk") == "Толке"
    assert app_label("unknown") == "" and app_locative("unknown") == ""


def test_paths_are_pinned_after_first_write(vault):
    cfg = Config(vault_root=vault)
    s = Session(vault / "s", meta(), State())
    p1 = write_raw(cfg, s, status_text="идёт запись")
    n1 = ensure_stub(cfg, s)
    s.meta.title = "Переименовали встречу на ходу"      # Толк меняет заголовок посреди записи
    p2 = write_raw(cfg, s, status_text="идёт запись")
    n2 = ensure_stub(cfg, s)
    assert p2 == p1 and n2 == n1
    assert len(list((vault / "Memory/50 · Raw/Local").iterdir())) == 1
    assert len(list((vault / "Memory/30 · Meetings").iterdir())) == 1


def test_n_speakers_counts_sys_speakers_without_embeddings(vault):
    cfg = Config(vault_root=vault)
    st = State(records=[Rec("sys", 0.5, 0.8, "Ага.", "Speaker 1", None),
                        Rec("sys", 1.0, 3.0, "Дальше.", "Speaker 2", 0),
                        Rec("mic", 3.5, 4.0, "Да.", "Me", None)],
               speaker_of_emb=[0])
    s = Session(vault / "s", meta(), st)
    assert "Говорящих на канале собеседников: 2" in write_raw(cfg, s, status_text="идёт запись").read_text()
