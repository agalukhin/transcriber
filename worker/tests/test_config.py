import importlib
from pathlib import Path

from transcriber import config as config_module
from transcriber.config import Config, load_config


def test_defaults_when_no_yaml(tmp_path):
    cfg = load_config(yaml_path=tmp_path / "missing.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.sample_rate == 16000
    assert cfg.min_silence_ms == 700
    assert cfg.max_utterance_s == 30.0
    assert cfg.glossary == []
    assert cfg.vault_root.name == "Вольт воркфлоу"


def test_yaml_and_glossary_override(tmp_path):
    (tmp_path / "c.yaml").write_text("vault_root: /tmp/vault\nmin_silence_ms: 500\nhallucinations: ['Спасибо за просмотр']\n")
    (tmp_path / "g.txt").write_text("CDP\n\nОмни\n")
    cfg = load_config(yaml_path=tmp_path / "c.yaml", glossary_path=tmp_path / "g.txt")
    assert cfg.vault_root == Path("/tmp/vault")
    assert cfg.min_silence_ms == 500
    assert cfg.hallucinations == ["Спасибо за просмотр"]
    assert cfg.glossary == ["CDP", "Омни"]


def test_unknown_yaml_keys_are_skipped_with_warning(tmp_path, caplog):
    (tmp_path / "c.yaml").write_text("min_silence_ms: 400\nnope: 1\n")
    with caplog.at_level("WARNING", logger="transcriber"):
        cfg = load_config(yaml_path=tmp_path / "c.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.min_silence_ms == 400
    assert "nope" in caplog.text


def test_root_can_be_overridden_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_ROOT", str(tmp_path))
    try:
        reloaded = importlib.reload(config_module)
        assert reloaded.ROOT == tmp_path
        assert reloaded.Config().sessions_dir == tmp_path / "sessions"
    finally:
        monkeypatch.delenv("TRANSCRIBER_ROOT")
        importlib.reload(config_module)
