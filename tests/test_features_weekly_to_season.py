import polars as pl
import pytest

from fantasy_ml.features.weekly_to_season import (
    season_totals_from_weekly,
    validate_weekly_schema,
    _sum_numeric_column_or_zero,
)

"""Unit tests for weekly-to-season feature aggregation helpers."""

def test_sum_numeric_column_or_zero_sums_present_values() -> None:
    """
    Sum numeric values for an existing column.

    This verifies that null values are treated as zero and that the
    aggregation returns the expected alias for each group.
    """
    df = pl.DataFrame(
        {
            "player_id": [1, 1, 2],
            "targets": [3, None, 5],
        }
    )

    result = (
        df.group_by("player_id")
        .agg(_sum_numeric_column_or_zero(df, "targets", "targets_sum"))
        .sort("player_id")
    )

    assert result.to_dicts() == [
        {"player_id": 1, "targets_sum": 3.0},
        {"player_id": 2, "targets_sum": 5.0},
    ]


def test_sum_numeric_column_or_zero_returns_zero_when_missing() -> None:
    """
    Return zero for each group when the source column is missing.

    This ensures the helper remains safe to use with partial schemas.
    """
    df = pl.DataFrame(
        {
            "player_id": [1, 1, 2],
        }
    )

    result = (
        df.group_by("player_id")
        .agg(_sum_numeric_column_or_zero(df, "targets", "targets_sum"))
        .sort("player_id")
    )

    assert result.to_dicts() == [
        {"player_id": 1, "targets_sum": 0.0},
        {"player_id": 2, "targets_sum": 0.0},
    ]
    
def test_validate_weekly_schema_requires_minimal_columns():
    df = pl.DataFrame({"player_id": ["x"]})
    with pytest.raises(ValueError):
        validate_weekly_schema(df)


def test_season_totals_from_weekly_sums_points_and_groups():
    df = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 1],
            "player_id": ["p1", "p1", "p2"],
            "player_display_name": ["A", "A", "B"],
            "position": ["RB", "RB", "RB"],
            "fp": [10.0, 5.0, 7.0],
            "targets": [2, 1, 0],
            "carries": [10, 12, 8],
            "attempts": [0, 0, 0],
        }
    )

    out = season_totals_from_weekly(df, points_col="fp", out_points_col="season_fp")

    # p1: 15, p2: 7
    assert out.filter(pl.col("player_id") == "p1")["season_fp"][0] == 15.0
    assert out.filter(pl.col("player_id") == "p2")["season_fp"][0] == 7.0

    assert "targets_sum" in out.columns
    assert "carries_sum" in out.columns
    assert "pass_attempts_sum" in out.columns


def test_season_totals_from_weekly_requires_points_col():
    df = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "player_id": ["p1"],
            "player_display_name": ["A"],
            "position": ["RB"],
        }
    )
    with pytest.raises(ValueError):
        season_totals_from_weekly(df, points_col="missing")