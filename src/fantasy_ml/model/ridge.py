from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RidgeResult:
    """Container for trained Ridge model and holdout evaluation metrics."""

    # Trained scikit-learn Ridge model instance.
    model: Ridge
    # Feature column names used during training.
    feature_cols: List[str]
    # Seasons included in the training split.
    train_seasons: List[int]
    # Seasons included in the test split.
    test_seasons: List[int]
    # Mean Absolute Error on holdout set (lower is better).
    mae: float
    # Root Mean Squared Error on holdout set (lower is better).
    rmse: float
    # R-squared on holdout set (higher is better, can be negative).
    r2: float


@dataclass(frozen=True)
class RegressionMetrics:
    """Stores MAE, RMSE, and R² for a set of predictions."""

    # Mean Absolute Error (lower is better).
    mae: float
    # Root Mean Squared Error (lower is better).
    rmse: float
    # R-squared (higher is better, can be negative).
    r2: float


@dataclass(frozen=True)
class RidgeVsNaiveResult:
    """Stores Ridge and naive baseline metrics on the same time split."""

    ridge_metrics: RegressionMetrics
    naive_metrics: RegressionMetrics
    # Prediction rows with y_true, y_pred_ridge, y_pred_naive columns.
    predictions: pl.DataFrame


@dataclass(frozen=True)
class PerPositionRidgeResult:
    """Stores per-position Ridge models, metrics, and overall metrics."""

    # One fitted Ridge model per position key.
    models_by_position: Dict[str, Ridge]
    # Per-position evaluation metrics.
    metrics_by_position: Dict[str, RegressionMetrics]
    # Overall metrics computed from concatenated per-position predictions.
    overall_metrics: RegressionMetrics
    # All prediction rows concatenated across positions.
    predictions: pl.DataFrame


def _to_numpy(df: pl.DataFrame, cols: Iterable[str]) -> np.ndarray:
    """
    Convert selected Polars columns into a NumPy matrix for scikit-learn.

    scikit-learn estimators expect NumPy arrays (or similar array-like input),
    so this helper keeps conversion logic in one place.
    """
    return df.select(list(cols)).to_numpy()


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """
    Compute MAE, RMSE, and R² from true and predicted arrays.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Predicted values from the model.

    Returns
    -------
    RegressionMetrics with mae, rmse, and r2 fields.

    Notes
    -----
    R² is set to NaN when fewer than 2 samples are present because the
    metric is undefined for a single data point.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    # mean_squared_error returns MSE; take sqrt manually to get RMSE.
    # Note: the `squared` parameter was removed in scikit-learn 1.4.
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    # r2_score requires at least 2 samples to be meaningful.
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)


def time_split(
    df: pl.DataFrame,
    *,
    season_col: str = "season",
    train_end_season: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split data by season to avoid future-data leakage.

    Train uses rows where season <= train_end_season.
    Test uses rows where season > train_end_season.
    """
    if season_col not in df.columns:
        raise ValueError(f"Missing column: {season_col!r}")

    # Time-aware split is preferred in forecasting tasks because random splits
    # can leak "future" information into training.
    train = df.filter(pl.col(season_col) <= train_end_season)
    test = df.filter(pl.col(season_col) > train_end_season)
    return train, test


def train_ridge(
    df: pl.DataFrame,
    *,
    feature_cols: List[str],
    target_col: str,
    season_col: str = "season",
    train_end_season: int,
    alpha: float = 1.0,
) -> RidgeResult:
    """
    Train Ridge regression and evaluate on a time-based holdout.

    Ridge regression is linear regression with L2 regularization. The `alpha`
    parameter controls regularization strength:
      - Larger alpha -> stronger shrinkage of coefficients
      - Smaller alpha -> behaves more like ordinary linear regression
    """
    # Validate that all required columns exist before any modeling work.
    for c in feature_cols + [target_col, season_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c!r}")

    train_df, test_df = time_split(
        df,
        season_col=season_col,
        train_end_season=train_end_season,
    )
    if train_df.is_empty() or test_df.is_empty():
        raise ValueError(
            "Train/test split produced an empty set; "
            "adjust train_end_season."
        )

    # Build design matrices (X) and target vectors (y).
    X_train = _to_numpy(train_df, feature_cols)
    y_train = train_df[target_col].to_numpy()

    X_test = _to_numpy(test_df, feature_cols)
    y_test = test_df[target_col].to_numpy()

    # Fit Ridge model.
    # random_state is included for reproducibility when solver paths use it.
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X_train, y_train)

    # Evaluate on holdout data (unseen future seasons).
    preds = model.predict(X_test)
    m = _compute_metrics(y_test, preds)

    # Keep explicit season lists for traceability in logs/reports.
    train_seasons = sorted(train_df[season_col].unique().to_list())
    test_seasons = sorted(test_df[season_col].unique().to_list())

    return RidgeResult(
        model=model,
        feature_cols=feature_cols,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        mae=m.mae,
        rmse=m.rmse,
        r2=m.r2,
    )


def predict_for_season(
    model: Ridge,
    df_features: pl.DataFrame,
    *,
    feature_cols: List[str],
    season: int,
    season_col: str = "season",
    out_col: str = "predicted_next_season_points",
) -> pl.DataFrame:
    """
    Generate predictions for one requested season.

    Returns the season subset with an added prediction column.
    """
    # Validate schema to avoid silent failures or cryptic model errors.
    for c in feature_cols + [season_col]:
        if c not in df_features.columns:
            raise ValueError(f"Missing required column: {c!r}")

    subset = df_features.filter(pl.col(season_col) == season)
    if subset.is_empty():
        raise ValueError(f"No rows found for season={season}")

    # Predict using the same feature columns used in training.
    X = _to_numpy(subset, feature_cols)
    preds = model.predict(X)

    # Attach predictions so downstream steps can join/export easily.
    return subset.with_columns(pl.Series(out_col, preds))


def compare_ridge_to_naive(
    df: pl.DataFrame,
    *,
    feature_cols: List[str],
    target_col: str,
    train_end_season: int,
    naive_source_col: str = "feat_points",
    season_col: str = "season",
    position_col: str = "position",
    alpha: float = 1.0,
) -> RidgeVsNaiveResult:
    """
    Compare Ridge regression to a naive baseline on the same time split.

    The naive baseline predicts next-season points as equal to current-season
    points (i.e. y_pred = feat_points). This is a useful sanity check:
    if Ridge cannot beat the naive baseline, the model is not adding value.

    Parameters
    ----------
    df:
        Full feature dataset with all seasons.
    feature_cols:
        Feature columns used by Ridge.
    target_col:
        Column holding the true next-season points.
    train_end_season:
        Last season included in training. Test is everything after.
    naive_source_col:
        Column used as the naive prediction (default: feat_points).
    season_col:
        Name of the season column.
    position_col:
        Name of the position column.
    alpha:
        Ridge regularization strength.
    """
    # Validate all columns up front before doing any expensive work.
    for c in feature_cols + [
        target_col, season_col, position_col, naive_source_col
    ]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c!r}")

    # Train Ridge on the same split used for naive comparison.
    ridge_result = train_ridge(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        season_col=season_col,
        train_end_season=train_end_season,
        alpha=alpha,
    )

    _, test_df = time_split(
        df,
        season_col=season_col,
        train_end_season=train_end_season,
    )

    X_test = _to_numpy(test_df, feature_cols)
    y_true = test_df[target_col].to_numpy()

    y_pred_ridge = ridge_result.model.predict(X_test)
    y_pred_naive = (
        test_df
        .select(pl.col(naive_source_col).cast(pl.Float64).fill_null(0.0))
        .to_series()
        .to_numpy()
    )

    # Build prediction rows for downstream inspection or plotting.
    pred_cols = [season_col, position_col]
    if "player_id" in test_df.columns:
        pred_cols.append("player_id")

    predictions = test_df.select(pred_cols).with_columns(
        [
            pl.Series("y_true", y_true),
            pl.Series("y_pred_ridge", y_pred_ridge),
            pl.Series("y_pred_naive", y_pred_naive),
        ]
    )

    return RidgeVsNaiveResult(
        ridge_metrics=_compute_metrics(y_true, y_pred_ridge),
        naive_metrics=_compute_metrics(y_true, y_pred_naive),
        predictions=predictions,
    )


def train_ridge_by_position(
    df: pl.DataFrame,
    *,
    feature_cols: List[str],
    target_col: str,
    train_end_season: int,
    season_col: str = "season",
    position_col: str = "position",
    positions: Sequence[str] = ("QB", "RB", "WR", "TE", "K"),
    alpha: float = 1.0,
) -> PerPositionRidgeResult:
    """
    Train one Ridge model per position and return per-position + overall metrics.

    Why train per position?
    Players at different positions score fantasy points in completely different
    ways (QBs throw touchdowns, RBs run the ball, Ks kick field goals). A
    single model trained on all positions at once would struggle to learn the
    unique scoring patterns of each. Training one model per position lets each
    model specialize.

    Overall metrics are computed from the concatenated per-position prediction
    rows so they reflect all positions together.

    Parameters
    ----------
    df:
        Full feature dataset with all seasons and positions.
    feature_cols:
        Feature columns to use as inputs for every position model.
    target_col:
        Column holding the true next-season points.
    train_end_season:
        Last season included in training. Test is everything after.
    season_col:
        Name of the season column.
    position_col:
        Name of the position column.
    positions:
        Which positions to train models for.
    alpha:
        Ridge regularization strength applied to all position models.
    """
    # Validate all required columns before looping over positions.
    for c in feature_cols + [target_col, season_col, position_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c!r}")

    models: Dict[str, Ridge] = {}
    metrics_by_position: Dict[str, RegressionMetrics] = {}
    prediction_frames: List[pl.DataFrame] = []

    for pos in positions:
        pos_df = df.filter(pl.col(position_col) == pos)

        # Skip positions that have no data in this dataset.
        if pos_df.is_empty():
            continue

        train_df, test_df = time_split(
            pos_df,
            season_col=season_col,
            train_end_season=train_end_season,
        )

        # Skip positions where the split leaves one side empty.
        if train_df.is_empty() or test_df.is_empty():
            continue

        X_train = _to_numpy(train_df, feature_cols)
        y_train = train_df[target_col].to_numpy()
        X_test = _to_numpy(test_df, feature_cols)
        y_true = test_df[target_col].to_numpy()

        model = Ridge(alpha=alpha, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        models[pos] = model
        metrics_by_position[pos] = _compute_metrics(y_true, y_pred)

        # Collect prediction rows for overall metric aggregation.
        pred_cols = [season_col, position_col]
        if "player_id" in test_df.columns:
            pred_cols.append("player_id")

        prediction_frames.append(
            test_df.select(pred_cols).with_columns(
                [
                    pl.Series("y_true", y_true),
                    pl.Series("y_pred", y_pred),
                ]
            )
        )

    if not prediction_frames:
        raise ValueError(
            "No position produced both train and test rows "
            "on the chosen split."
        )

    # Concatenate all position predictions to compute overall metrics.
    predictions = pl.concat(prediction_frames, how="vertical_relaxed")
    overall_metrics = _compute_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
    )

    return PerPositionRidgeResult(
        models_by_position=models,
        metrics_by_position=metrics_by_position,
        overall_metrics=overall_metrics,
        predictions=predictions,
    )