from __future__ import annotations

import polars as pl


ID_COLS = ["season", "player_id", "player_display_name", "position"]


def build_season_features(
    season_totals: pl.DataFrame,
    *,
    points_col: str = "season_fantasy_points",
) -> pl.DataFrame:
    """
    Turn season totals into a ML-ready feature table.

    Input: one row per (season, player_id, position) with season totals columns.
    Output: same rows plus engineered feature columns.

    This is intentionally "simple but sane" and purely deterministic.
    """
    missing = [c for c in ID_COLS if c not in season_totals.columns]
    if missing:
        raise ValueError(f"Missing required ID columns: {missing}")
    if points_col not in season_totals.columns:
        raise ValueError(f"Missing points column: {points_col!r}")

    df = season_totals

    # Ensure numeric columns exist
    def num_or_zero(col: str) -> pl.Expr:
        return (
            pl.col(col).cast(pl.Float64).fill_null(0.0)
            if col in df.columns
            else pl.lit(0.0)
        )

    games = num_or_zero("games_in_data")
    carries = num_or_zero("carries_sum")
    targets = num_or_zero("targets_sum")
    pass_att = num_or_zero("pass_attempts_sum")
    season_points = pl.col(points_col).cast(pl.Float64).fill_null(0.0)

    # Per-game rates (protect against divide by zero)
    games_safe = pl.when(games > 0).then(games).otherwise(pl.lit(1.0))

    return df.with_columns(
        [
            season_points.alias("feat_points"),
            (season_points / games_safe).alias("feat_ppg"),
            carries.alias("feat_carries"),
            (carries / games_safe).alias("feat_carries_pg"),
            targets.alias("feat_targets"),
            (targets / games_safe).alias("feat_targets_pg"),
            pass_att.alias("feat_pass_att"),
            (pass_att / games_safe).alias("feat_pass_att_pg"),
        ]
    )