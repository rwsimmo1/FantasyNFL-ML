from __future__ import annotations

import polars as pl


ID_COLS = ["season", "player_id", "player_display_name", "position"]


def build_season_features(
    season_totals: pl.DataFrame,
    *,
    points_col: str = "season_fantasy_points",
) -> pl.DataFrame:
    """
    Turn season totals into an ML-ready feature table.

    Input: one row per (season, player_id, position) with season totals columns.
    Output: same rows plus engineered feature columns.

    Age features (Pass 3 addition):
    --------------------------------
    feat_age    - player age during that season (null if not available)
    feat_age_sq - age squared, which captures the non-linear aging curve.

    Why age squared?
    Ridge regression fits a straight line through the data. But the
    relationship between age and performance is a curve (players improve,
    peak, then decline). Adding age² gives the model a "bent line" shape
    that can capture this curve. This technique is called adding a
    polynomial feature.

    Example:
        A 24-year-old RB might be improving  → age coeff is positive
        A 32-year-old RB is likely declining → age² coeff pulls score down
        Together they approximate the real aging curve.
    """
    missing = [c for c in ID_COLS if c not in season_totals.columns]
    if missing:
        raise ValueError(f"Missing required ID columns: {missing}")

    if points_col not in season_totals.columns:
        raise ValueError(f"Missing points column: {points_col!r}")

    df = season_totals

    def num_or_zero(col: str) -> pl.Expr:
        """Return column as Float64 (filling nulls with 0) or a zero literal."""
        if col not in df.columns:
            return pl.lit(0.0)
        return pl.col(col).cast(pl.Float64).fill_null(0.0)

    def num_or_null(col: str) -> pl.Expr:
        """
        Return column as Float64 preserving nulls, or a null literal.

        We use this for age rather than num_or_zero because filling a
        missing age with 0 would tell the model the player is 0 years old,
        which would badly distort the age feature. Keeping it null and
        letting Ridge handle missing values via imputation (or dropping
        those rows) is safer.
        """
        if col not in df.columns:
            return pl.lit(None).cast(pl.Float64)
        return pl.col(col).cast(pl.Float64)

    games      = num_or_zero("games_in_data")
    carries    = num_or_zero("carries_sum")
    targets    = num_or_zero("targets_sum")
    pass_att   = num_or_zero("pass_attempts_sum")
    season_points = pl.col(points_col).cast(pl.Float64).fill_null(0.0)

    # Age: preserve nulls so missing ages do not become misleading zeros.
    age = num_or_null("age")

    # Per-game rates (protect against divide-by-zero for players with 0 games).
    games_safe = pl.when(games > 0).then(games).otherwise(pl.lit(1.0))

    return df.with_columns(
        [
            season_points.alias("feat_points"),

            # --- Raw count features ---
            games.alias("feat_games"),
            carries.alias("feat_carries"),
            targets.alias("feat_targets"),
            pass_att.alias("feat_pass_att"),

            # --- Per-game rate features ---
            (season_points / games_safe).alias("feat_ppg"),
            (carries / games_safe).alias("feat_carries_pg"),
            (targets / games_safe).alias("feat_targets_pg"),
            (pass_att / games_safe).alias("feat_pass_att_pg"),

            # --- Age features (Pass 3 addition) ---
            # feat_age: raw age in years at time of the season.
            age.alias("feat_age"),

            # feat_age_sq: age squared, used to model the non-linear aging
            # curve. Without this, Ridge can only fit a straight line
            # through age vs performance, which misses the peak-and-decline
            # shape of real NFL careers.
            (age * age).alias("feat_age_sq"),
        ]
    )