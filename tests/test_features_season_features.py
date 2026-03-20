import polars as pl
import pytest

from fantasy_ml.features.season_features import build_season_features


def _base_df() -> pl.DataFrame:
    """
    Minimal valid input DataFrame used across multiple tests.

    Includes all required ID columns, a points column, and the optional
    source columns that feed into feature engineering.
    """
    return pl.DataFrame(
        {
            "season": [2025],
            "player_id": ["p1"],
            "player_display_name": ["A"],
            "position": ["RB"],
            "season_fantasy_points": [100.0],
            "games_in_data": [10],
            "carries_sum": [200],
            "targets_sum": [50],
            "pass_attempts_sum": [0],
            "age": [27],
        }
    )


def test_build_season_features_requires_id_cols():
    """Missing ID columns should raise ValueError immediately."""
    df = pl.DataFrame({"season": [2025]})
    with pytest.raises(ValueError):
        build_season_features(df)


def test_build_season_features_creates_expected_columns():
    """
    Verify that all expected feature columns are present in the output.

    Pass 1 additions:
      feat_games, feat_carries, feat_targets, feat_pass_att

    Pass 3 additions:
      feat_age    - raw player age
      feat_age_sq - age squared (captures non-linear aging curve)

    Existing per-game rate features:
      feat_ppg, feat_carries_pg, feat_targets_pg, feat_pass_att_pg
    """
    out = build_season_features(_base_df(), points_col="season_fantasy_points")

    for c in [
        "feat_points",
        "feat_games",
        "feat_carries",
        "feat_targets",
        "feat_pass_att",
        "feat_ppg",
        "feat_carries_pg",
        "feat_targets_pg",
        "feat_pass_att_pg",
        "feat_age",
        "feat_age_sq",
    ]:
        assert c in out.columns, f"Expected column {c!r} not found in output."


def test_build_season_features_age_values_are_correct():
    """
    feat_age should equal the input age and feat_age_sq = age².

    For age=27: feat_age=27.0, feat_age_sq=729.0
    """
    out = build_season_features(_base_df(), points_col="season_fantasy_points")

    assert out["feat_age"][0] == 27.0
    assert out["feat_age_sq"][0] == 729.0  # 27 * 27


def test_build_season_features_age_is_null_when_column_missing():
    """
    When the source data has no age column, feat_age and feat_age_sq
    should be null — NOT zero.

    Filling a missing age with 0 would tell the model the player is
    0 years old, which would corrupt the age feature entirely.
    """
    df = _base_df().drop("age")
    out = build_season_features(df, points_col="season_fantasy_points")

    # Both age features should be null, not 0.
    assert out["feat_age"][0] is None
    assert out["feat_age_sq"][0] is None


def test_build_season_features_spot_check_rate_features():
    """Spot-check per-game rate calculations."""
    out = build_season_features(_base_df(), points_col="season_fantasy_points")

    # 100 points / 10 games = 10.0 ppg
    assert out["feat_ppg"][0] == 10.0
    # 200 carries / 10 games = 20.0 carries per game
    assert out["feat_carries_pg"][0] == 20.0