"""Tests for Ridge-model helpers used in the fantasy NFL pipeline.

These tests are written for readers who are new to:
1) machine learning,
2) Python unit testing with pytest, and
3) Ridge regression.

The goal is to show *what behavior we expect* from each function.
"""

import polars as pl
import pytest

from fantasy_ml.model.ridge import (
    compare_ridge_to_naive,
    predict_for_season,
    time_split,
    train_ridge,
    train_ridge_by_position,
)


def _sample_training_frame() -> pl.DataFrame:
    """Create a tiny deterministic dataset for model tests.

    Why this helper exists:
    - Keeps test setup reusable.
    - Keeps each test focused on one behavior.
    - Small data makes expected outcomes easier to reason about.

    The dataset has two positions (RB and WR) and three seasons so that
    per-position training and time-aware splitting can both be exercised.
    """
    return pl.DataFrame(
        {
            # Seasons 2024 and 2025 are "history" for training.
            # Season 2026 is "future" for holdout testing.
            "season": [2024, 2024, 2025, 2025, 2026, 2026, 2024, 2025, 2026],
            "player_id": ["p1", "p2", "p1", "p2", "p1", "p2", "w1", "w1", "w1"],
            "position": ["RB", "RB", "RB", "RB", "RB", "RB", "WR", "WR", "WR"],
            # Example feature columns (inputs to model).
            "feat_points": [100.0, 90.0, 110.0, 95.0, 120.0, 98.0, 80.0, 85.0, 88.0],
            "feat_ppg": [10.0, 9.0, 11.0, 9.5, 12.0, 9.8, 8.0, 8.5, 8.8],
            "feat_carries_pg": [18.0, 16.0, 19.0, 16.5, 20.0, 17.0, 1.0, 1.1, 1.2],
            # Target column (what the model learns to predict).
            "target_next_season_points": [
                110.0, 95.0, 120.0, 98.0, 130.0, 100.0,
                85.0, 88.0, 90.0,
            ],
        }
    )


def test_time_split_partitions_rows_by_season() -> None:
    """Verify time-based split keeps past in train and future in test.

    ML concept:
    We split by time to avoid "future leakage" into training.
    """
    df = _sample_training_frame()

    train_df, test_df = time_split(
        df,
        season_col="season",
        train_end_season=2025,
    )

    # Train contains only seasons <= 2025.
    assert train_df["season"].max() <= 2025
    # Test contains only seasons > 2025.
    assert test_df["season"].min() > 2025


def test_train_ridge_returns_metrics_and_split_metadata() -> None:
    """Train a Ridge model and verify expected result fields exist.

    Unit-testing concept:
    We do not assert exact metric values (too brittle for tiny synthetic data).
    We assert structure/types and key expectations instead.
    """
    df = _sample_training_frame()

    result = train_ridge(
        df,
        feature_cols=["feat_ppg", "feat_carries_pg"],
        target_col="target_next_season_points",
        season_col="season",
        train_end_season=2025,
        alpha=1.0,  # Ridge regularization strength.
    )

    # Basic sanity checks for evaluation outputs.
    assert isinstance(result.mae, float)
    assert isinstance(result.rmse, float)
    assert isinstance(result.r2, float)

    # Verify split metadata is tracked for debugging/reporting.
    assert result.train_seasons == [2024, 2025]
    assert result.test_seasons == [2026]


def test_train_ridge_raises_when_split_is_empty() -> None:
    """Fail fast if split produces no test rows.

    This helps new users diagnose bad split choices quickly.
    """
    df = _sample_training_frame()

    # train_end_season=2026 leaves no seasons for test (> 2026), so error.
    with pytest.raises(ValueError, match="empty set"):
        train_ridge(
            df,
            feature_cols=["feat_ppg", "feat_carries_pg"],
            target_col="target_next_season_points",
            season_col="season",
            train_end_season=2026,
            alpha=1.0,
        )


def test_predict_for_season_adds_prediction_column() -> None:
    """Predict for one season and confirm output schema.

    Ridge concept - regression vs classification:

    There are two common types of ML prediction problems:

    1) REGRESSION - predicts a continuous number (any value on a scale).
       Example: "How many fantasy points will this player score next season?"
       Answer could be 47.3, 112.8, 205.0 - any number on a scale.
       Ridge regression is a regression model.

    2) CLASSIFICATION - predicts a category (a class label).
       Example: "Will this player finish in the top 10? Yes or No?"
       Answer is one of a fixed set of labels: "Yes" or "No".
       Ridge regression cannot do this.

    In this project we use Ridge regression to predict a continuous
    fantasy points total, NOT to classify players into categories.
    """
    df = _sample_training_frame()

    result = train_ridge(
        df,
        feature_cols=["feat_ppg", "feat_carries_pg"],
        target_col="target_next_season_points",
        season_col="season",
        train_end_season=2025,
        alpha=1.0,
    )

    out = predict_for_season(
        result.model,
        df,
        feature_cols=["feat_ppg", "feat_carries_pg"],
        season=2026,
        season_col="season",
        out_col="predicted_next_season_points",
    )

    # Output should contain predictions for the requested season rows.
    assert "predicted_next_season_points" in out.columns
    assert out.shape[0] >= 1


def test_predict_for_season_raises_for_missing_season() -> None:
    """Ensure clear error when requested season is not present."""
    df = _sample_training_frame()

    result = train_ridge(
        df,
        feature_cols=["feat_ppg", "feat_carries_pg"],
        target_col="target_next_season_points",
        season_col="season",
        train_end_season=2025,
        alpha=1.0,
    )

    with pytest.raises(ValueError, match="No rows found"):
        predict_for_season(
            result.model,
            df,
            feature_cols=["feat_ppg", "feat_carries_pg"],
            season=2030,
            season_col="season",
        )


def test_compare_ridge_to_naive_reports_both_metric_sets() -> None:
    """
    compare_ridge_to_naive should return metrics for both Ridge and baseline.

    ML concept - naive baseline:
    A naive baseline is the simplest possible prediction rule.  Here the
    baseline says "next season a player will score the same as this season."
    If our Ridge model cannot beat this baseline it is not adding value.

    Unit-testing concept:
    We check that the result has the right shape and types rather than
    asserting which model "wins" (that depends on the data).
    """
    df = _sample_training_frame()

    out = compare_ridge_to_naive(
        df,
        feature_cols=["feat_ppg", "feat_carries_pg"],
        target_col="target_next_season_points",
        train_end_season=2025,
        naive_source_col="feat_points",
    )

    # Both metric bundles should contain valid floats.
    assert isinstance(out.ridge_metrics.mae, float)
    assert isinstance(out.naive_metrics.mae, float)

    # Prediction rows should contain all three value columns.
    assert {"y_true", "y_pred_ridge", "y_pred_naive"}.issubset(
        set(out.predictions.columns)
    )


def test_train_ridge_by_position_returns_per_position_and_overall_metrics() -> None:
    """
    train_ridge_by_position should train one model per position and return
    both per-position metrics and combined overall metrics.

    ML concept - per-position models:
    QBs, RBs, WRs score fantasy points in completely different ways.
    Training a separate Ridge model for each position lets each model
    specialise on that position's scoring patterns.
    """
    df = _sample_training_frame()

    out = train_ridge_by_position(
        df,
        feature_cols=["feat_ppg", "feat_carries_pg"],
        target_col="target_next_season_points",
        train_end_season=2025,
        positions=("RB", "WR"),
    )

    # Both positions should have individual metric entries.
    assert set(out.metrics_by_position.keys()) == {"RB", "WR"}

    # Overall metrics are a float computed across all position predictions.
    assert isinstance(out.overall_metrics.mae, float)

    # Prediction rows should cover players from both positions.
    assert out.predictions.height >= 2