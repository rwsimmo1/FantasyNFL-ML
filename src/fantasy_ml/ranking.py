from __future__ import annotations

import polars as pl


POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def top_n_by_position(
    df: pl.DataFrame,
    *,
    position_col: str = "position",
    points_col: str = "fantasy_points",
    n: int = 24,
) -> pl.DataFrame:
    """
    Return top-N rows per position, sorted by fantasy points descending.

    Requirements:
    - df must include position_col and points_col
    - position is filtered to the canonical set (QB/RB/WR/TE/K/DST)

    Returns a DataFrame with the same columns as df.
    """
    if position_col not in df.columns:
        raise ValueError(f"Missing required column: {position_col}")
    if points_col not in df.columns:
        raise ValueError(f"Missing required column: {points_col}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    filtered = df.filter(pl.col(position_col).is_in(POSITIONS))

    # Rank within position, keep top n
    return (
        filtered.sort([position_col, points_col], descending=[False, True])
        .with_columns(
            pl.col(points_col).rank("dense", descending=True).over(position_col).alias(
                "_rank"
            )
        )
        .filter(pl.col("_rank") <= n)
        .drop("_rank")
        .sort([position_col, points_col], descending=[False, True])
    )