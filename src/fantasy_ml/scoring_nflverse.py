from __future__ import annotations

import polars as pl

from .scoring import FantasyScoring


def add_fantasy_points_nflverse_weekly(
    df: pl.DataFrame,
    scoring: FantasyScoring,
    *,
    out_col: str = "fantasy_points_calc",
) -> pl.DataFrame:
    """
    Compute fantasy points from nflverse weekly player stats schema.

    Important:
    - This function targets QB/RB/WR/TE/K player rows.
    - DST is NOT computed here (DST is a team-defense entity, handled separately).

    Expected nflverse columns (missing treated as 0):
      passing_yards, passing_tds, passing_interceptions
      rushing_yards, rushing_tds
      receptions, receiving_yards, receiving_tds
      sack_fumbles_lost, rushing_fumbles_lost, receiving_fumbles_lost
      fg_made, pat_made
    """
    # Helper: ensure a numeric column exists (float) and nulls become 0.
    def num(col: str) -> pl.Expr:
        if col not in df.columns:
            return pl.lit(0.0)
        return pl.col(col).cast(pl.Float64).fill_null(0.0)

    fumbles_lost = (
        num("sack_fumbles_lost")
        + num("rushing_fumbles_lost")
        + num("receiving_fumbles_lost")
    )

    return df.with_columns(
        (
            num("passing_yards") * scoring.pass_yd
            + num("passing_tds") * scoring.pass_td
            + num("passing_interceptions") * scoring.pass_int
            + num("rushing_yards") * scoring.rush_yd
            + num("rushing_tds") * scoring.rush_td
            + num("receptions") * scoring.rec
            + num("receiving_yards") * scoring.rec_yd
            + num("receiving_tds") * scoring.rec_td
            + fumbles_lost * scoring.fumble_lost
            + num("pat_made") * scoring.xpm
            + num("fg_made") * scoring.fgm
        ).alias(out_col)
    )