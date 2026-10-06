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


def test_clustering_and_echo_defaults_match_tuned_values(tmp_path):
    cfg = load_config(yaml_path=tmp_path / "missing.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.speaker_similarity == 0.3     # 0.5 давал 11 «спикеров» на созвон вчетвером
    assert cfg.min_embed_s == 1.0
    assert cfg.echo_strong_similarity == 0.8


def test_shipped_yaml_matches_tuned_defaults():
    import yaml as _yaml

    from transcriber.config import ROOT

    data = _yaml.safe_load((ROOT / "worker" / "config.yaml").read_text())
    assert data["speaker_similarity"] == 0.3
    assert data["echo_strong_similarity"] == 0.8


def test_block_defaults_and_shipped_yaml_agree(tmp_path):
    import yaml as _yaml

    from transcriber.config import ROOT

    cfg = load_config(yaml_path=tmp_path / "missing.yaml", glossary_path=tmp_path / "missing.txt")
    assert (cfg.block_min_s, cfg.block_max_s) == (45.0, 90.0)
    assert (cfg.block_gap_s, cfg.block_gap_keep_s, cfg.block_idle_s) == (1.5, 0.5, 4.0)
    assert cfg.carry_context is True
    data = _yaml.safe_load((ROOT / "worker" / "config.yaml").read_text())
    assert data["block_min_s"] == 45 and data["block_max_s"] == 90
    assert data["block_gap_s"] == 1.5 and data["carry_context"] is True


def test_voice_settings_have_defaults_and_expand_paths(tmp_path):
    cfg = load_config(yaml_path=tmp_path / "missing.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.voices_dir.name == "voices"
    assert cfg.voice_match_similarity == 0.45
    assert cfg.voice_min_samples == 3

    (tmp_path / "c.yaml").write_text("voices_dir: ~/голоса\nvoice_match_similarity: 0.6\nvoice_min_samples: 5\n")
    cfg = load_config(yaml_path=tmp_path / "c.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.voices_dir == Path("~/голоса").expanduser()
    assert cfg.voice_match_similarity == 0.6 and cfg.voice_min_samples == 5


def test_private_dir_has_a_default_and_is_expanded(tmp_path):
    cfg = load_config(yaml_path=tmp_path / "missing.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.private_dir == Path("~/Documents/Приватные записи").expanduser()
    assert cfg.private_dir.is_absolute()

    (tmp_path / "c.yaml").write_text("private_dir: ~/Приватное\n")
    cfg = load_config(yaml_path=tmp_path / "c.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.private_dir == Path("~/Приватное").expanduser()


def test_shipped_yaml_keeps_private_dir_outside_the_vault():
    import yaml as _yaml

    from transcriber.config import ROOT

    data = _yaml.safe_load((ROOT / "worker" / "config.yaml").read_text())
    assert "Приватные записи" in data["private_dir"]
    assert data["vault_root"] not in data["private_dir"]


def test_mlx_cache_default_and_shipped_yaml_agree(tmp_path):
    import yaml as _yaml

    from transcriber.config import ROOT

    cfg = load_config(yaml_path=tmp_path / "missing.yaml", glossary_path=tmp_path / "missing.txt")
    assert cfg.mlx_cache_mb == 2048
    data = _yaml.safe_load((ROOT / "worker" / "config.yaml").read_text())
    assert data["mlx_cache_mb"] == 2048


def test_memory_guards_default_and_shipped_yaml_agree(tmp_path):
    import yaml as _yaml

    from transcriber.config import ROOT

    cfg = load_config(yaml_path=tmp_path / "missing.yaml", glossary_path=tmp_path / "missing.txt")
    data = _yaml.safe_load((ROOT / "worker" / "config.yaml").read_text())
    assert data["idle_restart_mb"] == cfg.idle_restart_mb == 3500
    assert data["abandoned_after_min"] == cfg.abandoned_after_min == 10
