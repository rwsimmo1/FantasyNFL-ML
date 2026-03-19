"""
Unit tests for slice-based evaluation helpers in fantasy_ml.model.evaluation.

Testing philosophy for newcomers:
- Each test checks ONE behavior of ONE function (single responsibility).
- We use small hand-crafted DataFrames so expected outputs are easy to verify
  by reading the test itself.
- We do NOT use real NFL data in unit tests; that keeps tests fast and
  independent of data availability.
"""
import polars as pl
import pytest

from fantasy_ml.model.evaluation import (
    spearman_rank_by_position,
    top_n_overall_metrics,
    top_n_per_position_metrics,
)


def _pred_df() -> pl.DataFrame:
    """
    Small hand-crafted predictions DataFrame used across multiple tests.

    Each row represents one player's actual vs predicted next-season points
    in a single test season.
    """
    return pl.DataFrame(
        {
            "position": ["RB", "RB", "RB", "WR", "WR", "WR"],
            # y_true = what the player actually scored next season.
            "y_true": [120.0, 100.0, 80.0, 110.0, 90.0, 70.0],
            # y_pred = what the Ridge model predicted.
            "y_pred": [118.0, 95.0, 85.0, 105.0, 92.0, 68.0],
        }
    )


def test_top_n_per_position_metrics_returns_one_row_per_position() -> None:
    """
    top_n_per_position_metrics should return one row per position found.

    We ask for n=2 so each position keeps its two highest-scoring players.
    """
    out = top_n_per_position_metrics(_pred_df(), n=2)

    # One row per unique position in the input.
    assert out.height == 2
    # n column should reflect how many players were actually kept.
    assert out["n"].to_list() == [2, 2]
    # Expected output columns.
    assert set(out.columns) == {"position", "n", "mae", "rmse", "r2"}


def test_top_n_per_position_metrics_respects_n_limit() -> None:
    """When n is larger than the group, all rows in the group are kept."""
    out = top_n_per_position_metrics(_pred_df(), n=99)
    # Each position has 3 players so n should be 3, not 99.
    assert out["n"].to_list() == [3, 3]


def test_top_n_per_position_metrics_raises_on_missing_column() -> None:
    """Missing required column should raise ValueError immediately."""
    df = pl.DataFrame({"position": ["RB"], "y_true": [100.0]})  # y_pred missing
    with pytest.raises(ValueError, match="Missing required columns"):
        top_n_per_position_metrics(df)


def test_top_n_overall_metrics_returns_single_row() -> None:
    """
    top_n_overall_metrics should return exactly one summary row.

    We ask for n=3 so the top 3 players by actual points are evaluated.
    """
    out = top_n_overall_metrics(_pred_df(), n=3)

    assert out.height == 1
    assert out["n"][0] == 3
    assert set(out.columns) == {"n", "mae", "rmse", "r2"}


def test_top_n_overall_metrics_mae_is_non_negative() -> None:
    """MAE must always be >= 0 (it is an absolute value metric)."""
    out = top_n_overall_metrics(_pred_df(), n=6)
    assert out["mae"][0] >= 0.0


def test_top_n_overall_metrics_raises_on_missing_column() -> None:
    df = pl.DataFrame({"y_true": [100.0]})  # y_pred missing
    with pytest.raises(ValueError, match="Missing required columns"):
        top_n_overall_metrics(df)


def test_spearman_rank_by_position_returns_one_row_per_position() -> None:
    """
    spearman_rank_by_position should return one correlation row per position.

    Spearman = 1.0 would mean perfect rank ordering within that position.
    """
    out = spearman_rank_by_position(_pred_df())

    assert out.height == 2
    assert set(out.columns) == {"position", "n", "spearman"}


def test_spearman_rank_by_position_correlation_is_between_neg1_and_1() -> None:
    """Spearman correlation must always be in [-1, 1]."""
    out = spearman_rank_by_position(_pred_df())
    for val in out["spearman"].to_list():
        assert -1.0 <= val <= 1.0


def test_spearman_rank_by_position_perfect_rank_gives_1() -> None:
    """
    When predicted ranking exactly matches actual ranking, Spearman = 1.0.

    This is a useful sanity check: if we pass in y_pred == y_true the model
    has "perfect" rank knowledge and correlation should equal 1.0.
    """
    df = pl.DataFrame(
        {
            "position": ["RB", "RB", "RB"],
            "y_true": [120.0, 100.0, 80.0],
            # Identical to y_true so ranking is perfect.
            "y_pred": [120.0, 100.0, 80.0],
        }
    )
    out = spearman_rank_by_position(df)
    assert abs(out["spearman"][0] - 1.0) < 1e-9


def test_spearman_rank_by_position_raises_on_missing_column() -> None:
    df = pl.DataFrame({"position": ["RB"], "y_true": [100.0]})  # y_pred missing
    with pytest.raises(ValueError, match="Missing required columns"):
        spearman_rank_by_position(df)