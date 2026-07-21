import pytest

from soccer_highlights.config import Config, load_config


def test_load_config_defaults_when_no_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"
    cfg = load_config(missing_path)
    assert cfg == Config()


def test_load_config_reads_yaml_overrides(tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "detection:\n  strategy: onset_flux\ntimeline:\n  lookback_seconds: 30\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.detection.strategy == "onset_flux"
    assert cfg.timeline.lookback_seconds == 30


def test_load_config_unknown_key_raises(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("detection:\n  not_a_real_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_HL__DETECTION__STRATEGY", "onset_flux")
    monkeypatch.setenv("SOCCER_HL__TIMELINE__LOOKBACK_SECONDS", "12.5")
    missing_path = tmp_path / "does_not_exist.yaml"
    cfg = load_config(missing_path)
    assert cfg.detection.strategy == "onset_flux"
    assert cfg.timeline.lookback_seconds == 12.5
