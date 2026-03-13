from __future__ import annotations

import polars as pl


def validate_weekly_schema(df: pl.DataFrame) -> None:
    """Validate the minimal schema required for weekly-to-season aggregation."""
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
        raise ValueError(f"Weekly stats missing required columns: {missing}")


def _sum_numeric_column_or_zero(
    df: pl.DataFrame,
    column_name: str,
    alias_name: str,
) -> pl.Expr:
    """
    Build a safe aggregation expression for a numeric column.

    If the source column does not exist, a zero-valued literal is returned so
    the aggregation remains compatible with partial input schemas.
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
    df_weekly: pl.DataFrame,
    *,
    points_col: str,
    out_points_col: str = "season_fantasy_points",
) -> pl.DataFrame:
    """
    Aggregate weekly player rows into season totals.

    Parameters:
        df_weekly: Weekly player statistics.
        points_col: Name of the fantasy points column to aggregate.
        out_points_col: Output column name for aggregated season points.

    Returns:
        A DataFrame with one row per player-season-position grouping.
    """
    # Validate the required columns before any transformations are applied.
    validate_weekly_schema(df_weekly)

    # Fail early if the requested fantasy points column is missing.
    if points_col not in df_weekly.columns:
        raise ValueError(f"Missing points column: {points_col!r}")

    # Standardize fantasy points by treating null values as zero.
    df = df_weekly.with_columns(
        pl.col(points_col).cast(pl.Float64).fill_null(0.0)
    )

    # Aggregate season totals and usage metrics for downstream features.
    return (
        df.group_by(
            ["season", "player_id", "player_display_name", "position"]
        )
        .agg(
            [
                pl.col(points_col).sum().alias(out_points_col),
                pl.len().alias("games_in_data"),
                _sum_numeric_column_or_zero(df, "targets", "targets_sum"),
                _sum_numeric_column_or_zero(df, "carries", "carries_sum"),
                _sum_numeric_column_or_zero(
                    df,
                    "attempts",
                    "pass_attempts_sum",
                ),
            ]
        )
        .sort(
            ["season", "position", out_points_col],
            descending=[False, False, True],
        )
    )