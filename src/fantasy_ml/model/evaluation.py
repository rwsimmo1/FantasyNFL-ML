"""
Evaluation helpers for slice-based metrics on Ridge regression predictions.

These functions operate on a predictions DataFrame that must contain at minimum:
  - y_true:    actual next-season fantasy points
  - y_pred:    predicted next-season fantasy points
  - position:  player position (QB/RB/WR/TE/K)

They are pure functions (no side effects, no global state) which makes them
easy to unit test and reuse in notebooks, scripts, or runners.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _validate_cols(df: pl.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _metrics_dict(
    df: pl.DataFrame,
    actual_col: str,
    pred_col: str,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and R² from two columns in a DataFrame.

    ML note:
    - MAE:  average absolute error (same unit as points, easy to interpret)
    - RMSE: penalizes large errors more than MAE
    - R²:   fraction of variance explained (1.0 = perfect, 0.0 = mean baseline)
    """
    y_true = df[actual_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    mae = float(mean_absolute_error(y_true, y_pred))
    # sqrt(MSE) = RMSE; `squared` param removed in sklearn 1.4 so we do this manually.
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    # R² is undefined for a single sample so we guard against that case.
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def top_n_per_position_metrics(
    predictions: pl.DataFrame,
    *,
    n: int = 24,
    position_col: str = "position",
    actual_col: str = "y_true",
    pred_col: str = "y_pred",
) -> pl.DataFrame:
    """
    Compute MAE/RMSE/R² on the top-N players per position by actual points.

    Why top-N slices matter in fantasy:
    Fantasy managers care most about separating elite players from average ones.
    Evaluating only the top-24 RBs (for example) tells us how well the model
    ranks the players that actually get drafted and started.

    Returns one row per position with columns: position, n, mae, rmse, r2.
    """
    _validate_cols(predictions, [position_col, actual_col, pred_col])

    rows: list[dict] = []
    for pos in sorted(predictions[position_col].unique().to_list()):
        pos_df = predictions.filter(pl.col(position_col) == pos)
        # Keep only the top-N by actual points for this position.
        top_df = pos_df.sort(actual_col, descending=True).head(n)
        if top_df.is_empty():
            continue
        rows.append(
            {
                "position": pos,
                "n": int(top_df.height),
                **_metrics_dict(top_df, actual_col, pred_col),
            }
        )

    return pl.DataFrame(rows).sort("position") if rows else pl.DataFrame()


def top_n_overall_metrics(
    predictions: pl.DataFrame,
    *,
    n: int = 100,
    actual_col: str = "y_true",
    pred_col: str = "y_pred",
) -> pl.DataFrame:
    """
    Compute MAE/RMSE/R² on the top-N players overall by actual points.

    This gives an overall quality metric for the highest-value players
    across all positions combined (e.g. the top-100 draftable players).

    Returns a single-row DataFrame with columns: n, mae, rmse, r2.
    """
    _validate_cols(predictions, [actual_col, pred_col])
    top_df = predictions.sort(actual_col, descending=True).head(n)
    if top_df.is_empty():
        return pl.DataFrame()
    return pl.DataFrame([{"n": int(top_df.height), **_metrics_dict(top_df, actual_col, pred_col)}])


def spearman_rank_by_position(
    predictions: pl.DataFrame,
    *,
    position_col: str = "position",
    actual_col: str = "y_true",
    pred_col: str = "y_pred",
) -> pl.DataFrame:
    """
    Compute Spearman rank correlation between predicted and actual points
    within each position.

    What is Spearman correlation?
    Instead of comparing raw point values, Spearman ranks each player 1st,
    2nd, 3rd, etc. by actual points and by predicted points, then measures
    how well those rank orderings agree.
      - 1.0  = perfect rank agreement (model correctly orders all players)
      - 0.0  = no relationship between predicted and actual rank
      - -1.0 = model ranks players in completely reversed order

    In fantasy football, ranking players correctly (even if the raw point
    totals are off) is what separates a good draft from a bad one.

    Returns one row per position with columns: position, n, spearman.
    """
    _validate_cols(predictions, [position_col, actual_col, pred_col])

    rows: list[dict] = []
    for pos in sorted(predictions[position_col].unique().to_list()):
        pos_df = predictions.filter(pl.col(position_col) == pos)
        corr = _spearman_corr(
            pos_df[actual_col].to_numpy(),
            pos_df[pred_col].to_numpy(),
        )
        rows.append({"position": pos, "n": int(pos_df.height), "spearman": corr})

    return pl.DataFrame(rows).sort("position") if rows else pl.DataFrame()


def _spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between two arrays.

    Returns NaN when fewer than 2 samples are present or when either
    array has zero variance (all values identical).
    """
    if len(y_true) < 2:
        return float("nan")

    # Use Polars rank() so we stay consistent with the rest of the codebase.
    ranked = pl.DataFrame({"a": y_true, "b": y_pred}).with_columns(
        [
            pl.col("a").rank("average").alias("ra"),
            pl.col("b").rank("average").alias("rb"),
        ]
    )
    ra = ranked["ra"].to_numpy()
    rb = ranked["rb"].to_numpy()

    # Pearson correlation on ranks = Spearman correlation.
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])