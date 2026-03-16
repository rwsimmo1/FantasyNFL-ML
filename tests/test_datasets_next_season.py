import polars as pl
import pytest

from fantasy_ml.datasets.next_season import build_next_season_dataset


def test_build_next_season_dataset_joins_t_to_t_plus_1():
    df = pl.DataFrame(
        {
            "season": [2024, 2025, 2024],
            "player_id": ["p1", "p1", "p2"],
            "position": ["RB", "RB", "RB"],
            "feat_points": [100.0, 120.0, 50.0],
            "feat_ppg": [10.0, 12.0, 5.0],
        }
    )

    out = build_next_season_dataset(df, target_col="feat_points")
    print(out.head(3))
    # Only p1 has next-season (2024 -> 2025)
    assert out.shape[0] == 1
    assert out["season"][0] == 2024
    assert out["target_next_season_points"][0] == 120.0


def test_build_next_season_dataset_requires_columns():
    df = pl.DataFrame({"season": [2024]})
    with pytest.raises(ValueError):
        build_next_season_dataset(df)