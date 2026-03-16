from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    feature_cols: list[str]
    # Seasons included in the training split.
    train_seasons: list[int]
    # Seasons included in the test split.
    test_seasons: list[int]
    # Mean Absolute Error on holdout set (lower is better).
    mae: float
    # Root Mean Squared Error on holdout set (lower is better).
    rmse: float
    # R-squared on holdout set (higher is better, can be negative).
    r2: float


def _to_numpy(df: pl.DataFrame, cols: Iterable[str]) -> np.ndarray:
    """
    Convert selected Polars columns into a NumPy matrix for scikit-learn.

    scikit-learn estimators expect NumPy arrays (or similar array-like input),
    so this helper keeps conversion logic in one place.
    """
    return df.select(list(cols)).to_numpy()


def time_split(
    df: pl.DataFrame,
    *,
    season_col: str = "season",
    train_end_season: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    feature_cols: list[str],
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
    mae = float(mean_absolute_error(y_test, preds))

    # mean_squared_error returns MSE; take sqrt manually to get RMSE
    # Note: the `squared` parameter was removed in scikit-learn 1.4, so we compute RMSE 
    # manually with "** 0.5".
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    r2 = float(r2_score(y_test, preds))

    # Keep explicit season lists for traceability in logs/reports.
    train_seasons = sorted(train_df[season_col].unique().to_list())
    test_seasons = sorted(test_df[season_col].unique().to_list())

    return RidgeResult(
        model=model,
        feature_cols=feature_cols,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        mae=mae,
        rmse=rmse,
        r2=r2,
    )


def predict_for_season(
    model: Ridge,
    df_features: pl.DataFrame,
    *,
    feature_cols: list[str],
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