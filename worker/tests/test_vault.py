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
