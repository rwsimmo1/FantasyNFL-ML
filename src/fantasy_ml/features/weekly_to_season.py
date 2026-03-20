from __future__ import annotations

from typing import List

import polars as pl


def validate_weekly_schema(df: pl.DataFrame) -> None:
    """Validate the minimal schema required for weekly-to-season aggregation.

    Parameters
    ----------
    df : pl.DataFrame
        The weekly stats DataFrame to validate.

    Raises
    ------
    ValueError
        If any required column is missing from the DataFrame.
    """
    # Ensure the input contains the identifiers needed for grouping.
    required = [
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Weekly stats missing required columns: {missing}"
        )


def _sum_numeric_column_or_zero(
    df: pl.DataFrame,
    column_name: str,
    alias_name: str,
) -> pl.Expr:
    """Build a safe aggregation expression for a numeric column.

    If the source column does not exist, a zero-valued literal is
    returned so the aggregation remains compatible with partial input
    schemas.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame being aggregated (used to check column existence).
    column_name : str
        The source column to sum.
    alias_name : str
        The output column name for the aggregated result.

    Returns
    -------
    pl.Expr
        A Polars expression that sums the column or returns 0.0.
    """
    # Use a zero literal when the optional column is absent.
    if column_name not in df.columns:
        return pl.lit(0.0).alias(alias_name)

    # Sum the cleaned numeric values when the column is present.
    return (
        pl.col(column_name)
        .cast(pl.Float64)
        .fill_null(0.0)
        .sum()
        .alias(alias_name)
    )


def season_totals_from_weekly(
    weekly: pl.DataFrame,
    *,
    points_col: str = "fantasy_points_calc",
    out_points_col: str = "season_fantasy_points",
) -> pl.DataFrame:
    """Aggregate weekly player stats into season-level totals.

    Each row in the input represents one player's stats for one game.
    This function collapses all game rows for a (player, season) pair
    into a single summary row.

    Age handling
    ------------
    The nflreadpy weekly data does not include an age column — age is
    derived separately from the players roster table in age_features.py
    and joined onto the feature table in the pipeline runner. The age
    handling code below is kept for forward compatibility in case a
    future data source does include age in the weekly stats.

    Parameters
    ----------
    weekly : pl.DataFrame
        Weekly player stats with at least: season, player_id,
        player_display_name, position, week, and a points column.
    points_col : str
        Name of the column containing per-game fantasy points.
        Raises ValueError if this column is not present.
    out_points_col : str
        Name of the output column for season total fantasy points.

    Returns
    -------
    pl.DataFrame
        One row per (player_id, season) with summed stats and
        a season fantasy points total.

    Raises
    ------
    ValueError
        If required schema columns are missing, or if points_col
        is not found in the DataFrame.
    """
    # Validate required ID and grouping columns first so the caller
    # gets a clear error message rather than a cryptic Polars exception.
    validate_weekly_schema(weekly)

    # Validate the points column explicitly so callers get a clear
    # ValueError rather than a Polars ColumnNotFoundError buried inside
    # the aggregation call.
    if points_col not in weekly.columns:
        raise ValueError(
            f"Points column {points_col!r} not found in weekly data. "
            f"Available columns: {weekly.columns}"
        )

    # These are the columns we group by — one output row per combination.
    group_cols = ["season", "player_id", "player_display_name", "position"]

    # --- Build the aggregation expressions ---
    # Each expression below defines how to collapse multiple weekly rows
    # into a single season-total value.

    # Sum the fantasy points across all games in the season.
    agg_exprs: List[pl.Expr] = [
        pl.col(points_col).sum().alias(out_points_col),
    ]

    # Count how many games the player appeared in. This is used later
    # to compute per-game rate features (e.g. points per game).
    agg_exprs.append(
        pl.col("player_id").count().alias("games_in_data")
    )

    # Sum raw counting stats if they exist in the weekly data.
    # Using a helper avoids repeating the same null-check pattern.
    for raw_col, out_alias in [
        ("carries",  "carries_sum"),
        ("targets",  "targets_sum"),
        ("attempts", "pass_attempts_sum"),
    ]:
        if raw_col in weekly.columns:
            agg_exprs.append(
                pl.col(raw_col).sum().alias(out_alias)
            )

    # Age: take the max value across all weeks for this player-season.
    # Age is constant within a season so max == any non-null value.
    # We use max (rather than first/last) because it safely ignores
    # nulls — if any week has a valid age, we keep it.
    if "age" in weekly.columns:
        agg_exprs.append(
            pl.col("age").max().alias("age")
        )

    return weekly.group_by(group_cols).agg(agg_exprs)