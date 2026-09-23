import json
from transcriber.session import Rec, Session
from tests.conftest import make_session


def test_load_meta_and_fresh_state(tmp_path):
    d = make_session(tmp_path)
    s = Session.load(d)
    assert s.meta.id == "2026-09-15-1430"
    assert s.meta.app == "kontur.talk"
    assert s.meta.participants == ["Егор Спирин"]
    assert s.meta.started_at.hour == 14 and s.meta.started_at.minute == 30
    assert s.state.status == "recording"
    assert s.state.mic_offset == 0 and s.state.records == []
    assert s.mic_pcm == d / "mic.pcm" and s.sys_pcm == d / "sys.pcm"
    assert not s.stop_requested()


def test_state_roundtrip(tmp_path):
    d = make_session(tmp_path)
    s = Session.load(d)
    s.state.records.append(Rec(channel="sys", start=1.0, end=2.5, text="привет", speaker="Speaker 1", emb=0))
    s.state.embeddings.append([0.0, 1.0])
    s.state.speaker_of_emb.append(0)
    s.state.mic_offset = 16000
    s.save_state()
    assert not list(d.glob("*.tmp"))
    s2 = Session.load(d)
    assert s2.state.records[0] == Rec("sys", 1.0, 2.5, "привет", "Speaker 1", 0)
    assert s2.state.embeddings == [[0.0, 1.0]]
    assert s2.state.mic_offset == 16000
    raw = json.loads((d / "state.json").read_text())
    assert raw["records"][0]["text"] == "привет"


def test_stop_marker(tmp_path):
    d = make_session(tmp_path)
    (d / "STOP").touch()
    assert Session.load(d).stop_requested()


def test_rec_text_none_survives(tmp_path):
    d = make_session(tmp_path)
    s = Session.load(d)
    s.state.records.append(Rec("mic", 0.0, 1.0, None, "Me", None))
    s.save_state()
    assert Session.load(d).state.records[0].text is None


def test_unknown_keys_in_state_are_ignored(tmp_path):
    d = make_session(tmp_path)
    (d / "state.json").write_text(json.dumps({
        "status": "recording",
        "mic_offset": 512,
        "future": 1,
        "records": [{"channel": "sys", "start": 0.0, "end": 1.0, "text": "a", "speaker": "Speaker 1", "x": 1}],
        "pending_mic": [],
    }, ensure_ascii=False))
    s = Session.load(d)
    assert s.state.mic_offset == 512
    assert s.state.records == [Rec("sys", 0.0, 1.0, "a", "Speaker 1", None)]


def test_naive_started_at_gets_local_timezone(tmp_path):
    d = make_session(tmp_path)
    meta = json.loads((d / "meta.json").read_text())
    meta["started_at"] = "2026-09-15T14:30:00"
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    s = Session.load(d)
    assert s.meta.started_at.tzinfo is not None
    assert (s.meta.started_at.hour, s.meta.started_at.minute) == (14, 30)


def test_private_flag_is_parsed_and_defaults_to_false(tmp_path):
    assert Session.load(make_session(tmp_path)).meta.private is False
    d = make_session(tmp_path / "p", private=True)
    assert Session.load(d).meta.private is True


def test_private_flag_survives_a_non_boolean_value(tmp_path):
    d = make_session(tmp_path)
    meta = json.loads((d / "meta.json").read_text())
    meta["private"] = "true"          # Swift мог записать строкой — вольт всё равно бережём
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    assert Session.load(d).meta.private is True


def test_rec_kept_roundtrips_and_is_optional(tmp_path):
    d = make_session(tmp_path)
    s = Session.load(d)
    s.state.records.append(Rec("mic", 0.0, 1.0, "а почему?", "Me", None, kept=0.19))
    s.state.pending_mic.append(Rec("mic", 2.0, 3.0, "эхо", "Me", None, kept=0.07))
    s.state.records.append(Rec("mic", 4.0, 5.0, "без доли", "Me", None))
    s.save_state()
    s2 = Session.load(d)
    assert s2.state.records[0].kept == 0.19
    assert s2.state.pending_mic[0].kept == 0.07
    assert s2.state.records[1].kept is None


def test_rec_talk_roundtrips_and_is_optional(tmp_path):
    d = make_session(tmp_path)
    s = Session.load(d)
    s.state.records.append(Rec("sys", 0.0, 2.0, "да", "Speaker 1", 0, talk="Юлия Михайлова"))
    s.state.records.append(Rec("sys", 3.0, 4.0, "ага", "Speaker 1", 1))
    s.save_state()
    s2 = Session.load(d)
    assert s2.state.records[0].talk == "Юлия Михайлова"
    assert s2.state.records[1].talk is None
    assert json.loads((d / "state.json").read_text())["records"][0]["talk"] == "Юлия Михайлова"


def test_rec_without_talk_in_old_state_file(tmp_path):
    d = make_session(tmp_path)
    (d / "state.json").write_text(json.dumps({
        "records": [{"channel": "sys", "start": 0.0, "end": 1.0, "text": "да", "speaker": "Speaker 1", "emb": 0}],
    }))
    assert Session.load(d).state.records[0].talk is None


def test_rec_without_kept_in_old_state_file(tmp_path):
    d = make_session(tmp_path)
    (d / "state.json").write_text(json.dumps({
        "records": [{"channel": "mic", "start": 0.0, "end": 1.0, "text": "да", "speaker": "Me", "emb": None}],
    }))
    assert Session.load(d).state.records[0].kept is None
