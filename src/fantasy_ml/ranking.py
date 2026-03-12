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

    Parameters:
    df (pl.DataFrame):
        Input DataFrame containing player rows and scoring data.
    position_col (str):
        Column name that stores player positions. Defaults to "position".
    points_col (str):
        Column name that stores fantasy points. Defaults to "fantasy_points".
    n (int):
        Number of rows to keep per position. Must be a positive integer.

    Requirements:
    - df must include position_col and points_col
    - position is filtered to the canonical set (QB/RB/WR/TE/K/DST)

    Returns a DataFrame with the same columns as df.
    """
    # If the position column does not exist, stop early with a clear error.
    if position_col not in df.columns:
        raise ValueError(f"Missing required column: {position_col}")
    if points_col not in df.columns:
        raise ValueError(f"Missing required column: {points_col}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Keep only rows where position is one of the known fantasy positions.
    filtered = df.filter(pl.col(position_col).is_in(POSITIONS))

    # Rank within position, keep top n
    return (
        # Sort by position (A-Z), then by points (highest to lowest).
        filtered.sort([position_col, points_col], descending=[False, True])

        # Add a temporary rank column within each position group.
        # Highest points in each group get rank 1.
        .with_columns(
            pl.col(points_col).rank("dense", descending=True).over(position_col).alias(
                "_rank"
            )
        )

        # Keep only the top n ranks per position.
        .filter(pl.col("_rank") <= n)

        # Remove temporary rank column so output matches original schema.
        .drop("_rank")

        # Final sort for clean and predictable output ordering.
        .sort([position_col, points_col], descending=[False, True])
    )