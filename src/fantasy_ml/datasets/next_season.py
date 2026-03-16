from __future__ import annotations

import polars as pl


def build_next_season_dataset(
    features_by_season: pl.DataFrame,
    *,
    season_col: str = "season",
    player_id_col: str = "player_id",
    position_col: str = "position",
    target_col: str = "feat_points",
    out_target_col: str = "target_next_season_points",
) -> pl.DataFrame:
    """
    Build a supervised dataset where each row is (season t player) with target = points in season t+1.

    features_by_season should already contain engineered features (feat_*) and feat_points.
    """
    for c in (season_col, player_id_col, position_col, target_col):
        if c not in features_by_season.columns:
            raise ValueError(f"Missing required column: {c!r}")

    # Create a shifted copy to represent next season target
    next_season = features_by_season.select(
        [
            pl.col(season_col).alias("_season_next"),
            pl.col(player_id_col).alias("_player_id_next"),
            pl.col(position_col).alias("_position_next"),
            pl.col(target_col).alias(out_target_col),
        ]
    )

    # Join season t rows to season t+1 rows by player_id and position
    joined = features_by_season.join(
        next_season,
        left_on=[player_id_col, position_col, pl.col(season_col) + 1],
        right_on=["_player_id_next", "_position_next", "_season_next"],
        how="inner",
    ).drop(["_season_next", "_player_id_next", "_position_next"])

    return joined