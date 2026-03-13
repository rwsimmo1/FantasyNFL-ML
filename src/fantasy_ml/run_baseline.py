from __future__ import annotations

import argparse
import polars as pl

from fantasy_ml.config import load_config
from fantasy_ml.data_sources.base import WeeklyPlayerStatsSource
from fantasy_ml.data_sources.nflverse_source import NflverseWeeklyPlayerStatsSource
from fantasy_ml.features.weekly_to_season import season_totals_from_weekly
from fantasy_ml.ranking import top_n_by_position
from fantasy_ml.scoring import scoring_from_mode
from fantasy_ml.scoring_nflverse import add_fantasy_points_nflverse_weekly


def build_baseline_top24(
    source: WeeklyPlayerStatsSource,
    *,
    season: int,
    top_n: int = 24,
) -> pl.DataFrame:
    """
    Core baseline pipeline. Kept as a function to be testable.

    Returns:
      DataFrame of top-N by position for the given season.
    """
    cfg = load_config()
    scoring = scoring_from_mode(cfg.fantasy_scoring)

    weekly = source.load_weekly_player_stats(season)

    # Filter to the positions we can compute from player rows for now.
    # DST will be added later from a team-defense dataset.
    weekly = weekly.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"]))

    weekly_scored = add_fantasy_points_nflverse_weekly(
        weekly,
        scoring,
        out_col="fantasy_points_calc",
    )

    season_totals = season_totals_from_weekly(
        weekly_scored, points_col="fantasy_points_calc", out_points_col="season_points"
    )

    # rank using season totals
    return top_n_by_position(
        season_totals.rename({"season_points": "fantasy_points"}),
        n=top_n,
        points_col="fantasy_points",
        position_col="position",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    source = NflverseWeeklyPlayerStatsSource()
    out = build_baseline_top24(source, season=args.season, top_n=args.top_n)

    if args.out:
        out.write_csv(args.out)
        print(f"Wrote {args.out}")
    else:
        # Pretty-ish print to console
        print(out)


if __name__ == "__main__":
    main()