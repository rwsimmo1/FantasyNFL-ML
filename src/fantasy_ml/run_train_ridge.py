"""This script:

1. loads weekly stats for a range of seasons (e.g., 2016–2025)
2. scores weekly rows → aggregates to season totals → builds features
3. builds next-season supervised dataset
4. trains Ridge with time split
5. outputs 2026 predictions from 2025 rows
"""

from __future__ import annotations

import argparse
import polars as pl

from fantasy_ml.config import load_config
from fantasy_ml.data_sources.base import WeeklyPlayerStatsSource
from fantasy_ml.data_sources.nflverse_source import NflverseWeeklyPlayerStatsSource
from fantasy_ml.datasets.next_season import build_next_season_dataset
from fantasy_ml.features.season_features import build_season_features
from fantasy_ml.features.weekly_to_season import season_totals_from_weekly
from fantasy_ml.model.ridge import predict_for_season, train_ridge
from fantasy_ml.scoring import scoring_from_mode
from fantasy_ml.scoring_nflverse import add_fantasy_points_nflverse_weekly


def load_and_build_features(
    source: WeeklyPlayerStatsSource,
    *,
    seasons: list[int],
) -> pl.DataFrame:
    cfg = load_config()
    scoring = scoring_from_mode(cfg.fantasy_scoring)

    frames: list[pl.DataFrame] = []
    for s in seasons:
        weekly = source.load_weekly_player_stats(s)
        weekly = weekly.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"]))
        weekly_scored = add_fantasy_points_nflverse_weekly(
            weekly, scoring, out_col="fantasy_points_calc"
        )
        season_totals = season_totals_from_weekly(
            weekly_scored,
            points_col="fantasy_points_calc",
            out_points_col="season_fantasy_points",
        )
        frames.append(season_totals)

    all_seasons = pl.concat(frames, how="diagonal")
    return build_season_features(all_seasons, points_col="season_fantasy_points")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--train-end-season", type=int, default=2022)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out-preds", type=str, default="pred_2026_from_2025.csv")
    args = parser.parse_args()

    seasons = list(range(args.start_season, args.end_season + 1))
    source = NflverseWeeklyPlayerStatsSource()

    features = load_and_build_features(source, seasons=seasons)

    supervised = build_next_season_dataset(
        features,
        target_col="feat_points",
        out_target_col="target_next_season_points",
    )

    feature_cols = [
        "feat_ppg",
        "feat_carries_pg",
        "feat_targets_pg",
        "feat_pass_att_pg",
    ]

    res = train_ridge(
        supervised,
        feature_cols=feature_cols,
        target_col="target_next_season_points",
        train_end_season=args.train_end_season,
        alpha=args.alpha,
    )

    print("Train seasons:", res.train_seasons)
    print("Test seasons:", res.test_seasons)
    print("MAE:", res.mae)
    print("RMSE:", res.rmse)
    print("R2:", res.r2)

    preds_2026 = predict_for_season(
        res.model,
        features,
        feature_cols=feature_cols,
        season=2025,
        out_col="predicted_2026_points",
    ).select(
        [
            "season",
            "player_id",
            "player_display_name",
            "position",
            "feat_points",
            "predicted_2026_points",
        ]
    )

    preds_2026.write_csv(args.out_preds)
    print(f"Wrote {args.out_preds}")


if __name__ == "__main__":
    main()