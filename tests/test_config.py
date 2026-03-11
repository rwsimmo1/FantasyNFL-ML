import os
import pytest

from fantasy_ml.config import load_config


def test_load_config_defaults_to_ppr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FANTASY_SCORING", raising=False)
    monkeypatch.delenv("SCORING_PROVIDER", raising=False)
    cfg = load_config(dotenv_path=None)
    assert cfg.fantasy_scoring == "PPR"
    assert cfg.scoring_provider == "ESPN"


@pytest.mark.parametrize("val", ["PPR", "half_ppr", " Standard "])
def test_load_config_accepts_valid_values(monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("FANTASY_SCORING", val)
    cfg = load_config(dotenv_path=None)
    assert cfg.fantasy_scoring in ("PPR", "HALF_PPR", "STANDARD")


@pytest.mark.parametrize("val", ["ESPN", "yahoo", " Yahoo "])
def test_load_config_accepts_valid_provider(
    monkeypatch: pytest.MonkeyPatch,
    val: str,
):
    monkeypatch.setenv("SCORING_PROVIDER", val)
    cfg = load_config(dotenv_path=None)
    assert cfg.scoring_provider in ("ESPN", "YAHOO")


def test_load_config_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FANTASY_SCORING", "weird")
    with pytest.raises(ValueError):
        load_config(dotenv_path=None)


def test_load_config_rejects_invalid_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCORING_PROVIDER", "nfl")
    with pytest.raises(ValueError):
        load_config(dotenv_path=None)