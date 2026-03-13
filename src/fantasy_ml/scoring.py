from __future__ import annotations

from dataclasses import dataclass
import polars as pl

from fantasy_ml.scoring_rules import get_scoring_rules


DEFAULT_SCORING_RULES = get_scoring_rules("PPR", "ESPN")


# The dataclass decorator auto-generates __init__, __repr__, and value-based equality
# functions from the typed fields below (frozen=True makes instances immutable)
# for our FantasyScoring class.
@dataclass(frozen=True)
class FantasyScoring:
    # Passing
    pass_yd: float = DEFAULT_SCORING_RULES["pass_yd"]
    pass_td: float = DEFAULT_SCORING_RULES["pass_td"]
    pass_int: float = DEFAULT_SCORING_RULES["pass_int"]

    # Rushing
    rush_yd: float = DEFAULT_SCORING_RULES["rush_yd"]
    rush_td: float = DEFAULT_SCORING_RULES["rush_td"]

    # Receiving
    rec: float = DEFAULT_SCORING_RULES["rec"]
    rec_yd: float = DEFAULT_SCORING_RULES["rec_yd"]
    rec_td: float = DEFAULT_SCORING_RULES["rec_td"]

    # Misc
    fumble_lost: float = DEFAULT_SCORING_RULES["fumble_lost"]

    # DST (MVP baseline)
    def_sack: float = DEFAULT_SCORING_RULES["def_sack"]
    def_int: float = DEFAULT_SCORING_RULES["def_int"]
    def_fumble_recovery: float = DEFAULT_SCORING_RULES["def_fumble_recovery"]
    def_td: float = DEFAULT_SCORING_RULES["def_td"]

     # Kicking (MVP)
    xpm: float = 1.0
    fgm: float = 3.0


def scoring_from_mode(mode: str, provider: str = "ESPN") -> FantasyScoring:
    """Return ``FantasyScoring`` preset from scoring mode."""
    return FantasyScoring(**get_scoring_rules(mode=mode, provider=provider))


def add_fantasy_points(
    df: pl.DataFrame,
    scoring: FantasyScoring,
    *,
    pass_yd_col: str = "passing_yards",
    pass_td_col: str = "passing_tds",
    int_col: str = "interceptions",
    rush_yd_col: str = "rushing_yards",
    rush_td_col: str = "rushing_tds",
    rec_col: str = "receptions",
    rec_yd_col: str = "receiving_yards",
    rec_td_col: str = "receiving_tds",
    fumble_lost_col: str = "fumbles_lost",
    def_sack_col: str = "def_sacks",
    def_int_col: str = "def_interceptions",
    def_fumble_recovery_col: str = "def_fumble_recoveries",
    def_td_col: str = "def_tds",
    out_col: str = "fantasy_points",
) -> pl.DataFrame:
    """
    Adds a fantasy_points column. Missing columns are treated as zeros
    (helpful because different positions have different stat columns).

    This is deliberately pure and unit-testable.
    """
    def z(col: str) -> pl.Expr:
        return pl.when(pl.col(col).is_null()).then(0).otherwise(pl.col(col)).fill_null(0)

    required = [
        pass_yd_col, pass_td_col, int_col,
        rush_yd_col, rush_td_col,
        rec_col, rec_yd_col, rec_td_col,
        fumble_lost_col,
        def_sack_col, def_int_col, def_fumble_recovery_col, def_td_col,
    ]
    # If some cols don't exist, create them as 0
    for c in required:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).cast(pl.Float64).alias(c))

    return df.with_columns(
        (
            z(pass_yd_col) * scoring.pass_yd
            + z(pass_td_col) * scoring.pass_td
            + z(int_col) * scoring.pass_int
            + z(rush_yd_col) * scoring.rush_yd
            + z(rush_td_col) * scoring.rush_td
            + z(rec_col) * scoring.rec
            + z(rec_yd_col) * scoring.rec_yd
            + z(rec_td_col) * scoring.rec_td
            + z(fumble_lost_col) * scoring.fumble_lost
            + z(def_sack_col) * scoring.def_sack
            + z(def_int_col) * scoring.def_int
            + z(def_fumble_recovery_col) * scoring.def_fumble_recovery
            + z(def_td_col) * scoring.def_td
        ).alias(out_col)
    )