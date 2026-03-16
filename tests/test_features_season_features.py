import polars as pl
import pytest

from fantasy_ml.features.season_features import build_season_features


def test_build_season_features_requires_id_cols():
    df = pl.DataFrame({"season": [2025]})
    with pytest.raises(ValueError):
        build_season_features(df)


def test_build_season_features_creates_expected_columns():
    df = pl.DataFrame(
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
        }
    )

    out = build_season_features(df, points_col="season_fantasy_points")
    for c in [
        "feat_points",
        "feat_ppg",
        "feat_carries",
        "feat_carries_pg",
        "feat_targets",
        "feat_targets_pg",
        "feat_pass_att",
        "feat_pass_att_pg",
    ]:
        assert c in out.columns

    assert out["feat_ppg"][0] == 10.0
    assert out["feat_carries_pg"][0] == 20.0