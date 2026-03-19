"""Entry-point script for training a Ridge regression model on NFL fantasy data.

This script ties together the full data pipeline in six stages:

1. Load weekly player stats for a range of seasons (e.g. 2016-2025).
2. Score each weekly row and aggregate to season totals.
3. Build engineered feature columns from the season totals.
4. Build a supervised dataset where each row is:
       (player, season t) -> target = fantasy points in season t+1
5. Train Ridge regression using a time-aware split, then evaluate:
       - Overall Ridge metrics (MAE, RMSE, R²)
       - Ridge vs naive baseline comparison
       - Per-position Ridge models
       - Top-N slice metrics (top-24 per position, top-100 overall)
       - Spearman rank correlation by position
6. Generate 2026 predictions from each player's 2025 season stats.

How to run:
    python -m fantasy_ml.run_train_ridge
    python -m fantasy_ml.run_train_ridge --start-season 2018 --end-season 2025

See argparse arguments below for all available options.
"""

from __future__ import annotations

import argparse

import polars as pl

from fantasy_ml.config import load_config
from fantasy_ml.data_sources.base import WeeklyPlayerStatsSource
from fantasy_ml.data_sources.nflverse_source import (
    NflverseWeeklyPlayerStatsSource,
)
from fantasy_ml.datasets.next_season import build_next_season_dataset
from fantasy_ml.features.season_features import build_season_features
from fantasy_ml.features.weekly_to_season import season_totals_from_weekly
from fantasy_ml.model.evaluation import (
    spearman_rank_by_position,
    top_n_overall_metrics,
    top_n_per_position_metrics,
)
from fantasy_ml.model.ridge import (
    compare_ridge_to_naive,
    predict_for_season,
    train_ridge,
    train_ridge_by_position,
)
from fantasy_ml.scoring import scoring_from_mode
from fantasy_ml.scoring_nflverse import add_fantasy_points_nflverse_weekly


def load_and_build_features(
    source: WeeklyPlayerStatsSource,
    *,
    seasons: list[int],
) -> pl.DataFrame:
    """Load weekly stats for each season and build the ML feature table.

    This function isolates data loading and feature engineering so the
    rest of main() can focus on model training. Keeping IO separate from
    computation makes the pipeline easier to unit test.

    Parameters
    ----------
    source:
        A data-source object that knows how to fetch weekly player stats.
        Using an abstract base here (WeeklyPlayerStatsSource) means we can
        swap in a different data provider without changing this function.
    seasons:
        List of NFL seasons to load (e.g. [2016, 2017, ..., 2025]).

    Returns
    -------
    pl.DataFrame
        One row per (player, season) with all feat_* columns attached.
    """
    # Load the scoring weights (PPR, half-PPR, etc.) from project config.
    cfg = load_config()
    scoring = scoring_from_mode(cfg.fantasy_scoring)

    frames: list[pl.DataFrame] = []

    for s in seasons:
        # --- Step 1: Load raw weekly stats for one season ---
        weekly = source.load_weekly_player_stats(s)

        # Keep only the five fantasy-relevant positions.
        # Positions like FB, OL, or DL do not score fantasy points.
        weekly = weekly.filter(
            pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
        )

        # --- Step 2: Score each week ---
        # add_fantasy_points_nflverse_weekly applies the scoring weights
        # (e.g. 1 pt per reception in PPR) to produce a points column.
        weekly_scored = add_fantasy_points_nflverse_weekly(
            weekly,
            scoring,
            out_col="fantasy_points_calc",
        )

        # --- Step 3: Aggregate weekly rows into season totals ---
        # One game = one weekly row, so this collapses ~17-18 rows per
        # player into a single season-total row.
        season_totals = season_totals_from_weekly(
            weekly_scored,
            points_col="fantasy_points_calc",
            out_points_col="season_fantasy_points",
        )

        frames.append(season_totals)

    # Stack all seasons into one DataFrame.
    # diagonal_relaxed allows columns that exist in some seasons but not
    # others (e.g. a new stat column added in a later season).
    all_seasons = pl.concat(frames, how="diagonal")

    # --- Step 4: Engineer features ---
    # build_season_features adds feat_ppg, feat_carries_pg, etc. from the
    # raw season totals. These become the X columns for Ridge regression.
    return build_season_features(
        all_seasons,
        points_col="season_fantasy_points",
    )


def _print_section(title: str) -> None:
    """Print a visible section header to make console output easy to scan."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    """Parse CLI arguments, run the full pipeline, and write predictions.

    Pipeline overview
    -----------------
    1. Load + feature-engineer all seasons via load_and_build_features.
    2. Build the supervised (X, y) dataset.
    3. Train overall Ridge and print holdout metrics.
    4. Compare Ridge to naive baseline on the same split.
    5. Train per-position Ridge models and print per-position metrics.
    6. Print top-N slice metrics and Spearman rank correlations.
    7. Generate 2026 predictions from 2025 features and write to CSV.
    """
    # ----------------------------------------------------------------
    # CLI argument definitions
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train Ridge and generate next-season fantasy predictions."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2016,
        help="First NFL season to load (default: 2016).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2025,
        help="Last NFL season to load (default: 2025).",
    )
    # train_end_season controls the time-aware split.
    # Seasons <= this value go into training; later seasons go into test.
    # Example: train_end=2022 means 2023/2024/2025 are holdout seasons.
    parser.add_argument(
        "--train-end-season",
        type=int,
        default=2022,
        help=(
            "Last season included in Ridge training. "
            "Seasons after this are the holdout test set (default: 2022)."
        ),
    )
    # alpha is the Ridge regularization strength.
    # Higher alpha = stronger penalty on large coefficients = simpler model.
    # Lower alpha = regression behaves more like ordinary least squares.
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (default: 1.0).",
    )
    parser.add_argument(
        "--out-preds",
        type=str,
        default="pred_2026_from_2025.csv",
        help="Output CSV path for 2026 predictions (default: pred_2026_from_2025.csv).",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Stage 1-3: Load data and build the feature table
    # ----------------------------------------------------------------
    seasons = list(range(args.start_season, args.end_season + 1))
    source = NflverseWeeklyPlayerStatsSource()
    features = load_and_build_features(source, seasons=seasons)

    # ----------------------------------------------------------------
    # Stage 4: Build the supervised (X, y) dataset
    # ----------------------------------------------------------------
    # build_next_season_dataset joins each player's season-t features to
    # their season-(t+1) points total. Only players present in both
    # consecutive seasons are kept (inner join).
    supervised = build_next_season_dataset(
        features,
        target_col="feat_points",
        out_target_col="target_next_season_points",
    )

    # These are the input columns Ridge uses to make predictions (X matrix).
    # Per-game rates are used so players who missed games are not penalised
    # simply for playing fewer games.
    feature_cols = [
        "feat_ppg",           # average fantasy points per game
        "feat_carries_pg",    # average carries per game   (RB usage)
        "feat_targets_pg",    # average targets per game   (WR/TE usage)
        "feat_pass_att_pg",   # average pass attempts per game (QB usage)
    ]

    # ----------------------------------------------------------------
    # Stage 5a: Overall Ridge — train and print holdout metrics
    # ----------------------------------------------------------------
    _print_section("Overall Ridge — holdout metrics")

    overall = train_ridge(
        supervised,
        feature_cols=feature_cols,
        target_col="target_next_season_points",
        train_end_season=args.train_end_season,
        alpha=args.alpha,
    )

    print(f"Train seasons : {overall.train_seasons}")
    print(f"Test seasons  : {overall.test_seasons}")
    # MAE: average absolute error in fantasy points (lower is better).
    print(f"MAE           : {overall.mae:.2f}")
    # RMSE: penalises large errors more than MAE (lower is better).
    print(f"RMSE          : {overall.rmse:.2f}")
    # R²: fraction of variance explained (1.0 = perfect, 0.0 = mean baseline).
    print(f"R²            : {overall.r2:.4f}")

    # ----------------------------------------------------------------
    # Stage 5b: Ridge vs naive baseline
    # ----------------------------------------------------------------
    # The naive baseline predicts "next season = this season's points".
    # If Ridge cannot beat this, it is not adding value.
    _print_section("Ridge vs Naive baseline")

    vs_naive = compare_ridge_to_naive(
        supervised,
        feature_cols=feature_cols,
        target_col="target_next_season_points",
        train_end_season=args.train_end_season,
        naive_source_col="feat_points",
        alpha=args.alpha,
    )

    print(f"{'Metric':<10} {'Ridge':>10} {'Naive':>10} {'Winner':>10}")
    print("-" * 44)
    for metric in ("mae", "rmse", "r2"):
        ridge_val = getattr(vs_naive.ridge_metrics, metric)
        naive_val = getattr(vs_naive.naive_metrics, metric)
        # For MAE and RMSE lower is better; for R² higher is better.
        if metric == "r2":
            winner = "Ridge" if ridge_val > naive_val else "Naive"
        else:
            winner = "Ridge" if ridge_val < naive_val else "Naive"
        print(f"{metric.upper():<10} {ridge_val:>10.2f} {naive_val:>10.2f} {winner:>10}")

    # ----------------------------------------------------------------
    # Stage 5c: Per-position Ridge models
    # ----------------------------------------------------------------
    # One model per position so each can specialise on that position's
    # unique scoring patterns (QB passing vs RB rushing vs K kicking).
    _print_section("Per-position Ridge models")

    by_pos = train_ridge_by_position(
        supervised,
        feature_cols=feature_cols,
        target_col="target_next_season_points",
        train_end_season=args.train_end_season,
        alpha=args.alpha,
    )

    print(f"{'Position':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 40)
    for pos, m in sorted(by_pos.metrics_by_position.items()):
        print(f"{pos:<12} {m.mae:>8.2f} {m.rmse:>8.2f} {m.r2:>8.4f}")
    print("-" * 40)
    om = by_pos.overall_metrics
    print(f"{'Overall':<12} {om.mae:>8.2f} {om.rmse:>8.2f} {om.r2:>8.4f}")

    # ----------------------------------------------------------------
    # Stage 5d: Top-N slice metrics
    # ----------------------------------------------------------------
    # Evaluate only on the players that matter most in fantasy drafts.
    # Overall metrics include bench-warmers who score near zero and can
    # make the model look better than it really is for drafted players.
    _print_section("Top-24 per position — slice metrics")

    top24 = top_n_per_position_metrics(
        by_pos.predictions,
        n=24,
        actual_col="y_true",
        pred_col="y_pred",
    )
    print(top24)

    _print_section("Top-100 overall — slice metrics")

    top100 = top_n_overall_metrics(
        by_pos.predictions,
        n=100,
        actual_col="y_true",
        pred_col="y_pred",
    )
    print(top100)

    # ----------------------------------------------------------------
    # Stage 5e: Spearman rank correlation by position
    # ----------------------------------------------------------------
    # Measures whether the model ranks players in the right order within
    # each position, independent of whether the raw point totals are accurate.
    # In fantasy, ranking correctly is often more important than exact totals.
    _print_section("Spearman rank correlation by position")

    spearman = spearman_rank_by_position(
        by_pos.predictions,
        actual_col="y_true",
        pred_col="y_pred",
    )
    print(spearman)
    print()
    print("Interpreting Spearman:")
    print("  > 0.7  strong rank agreement")
    print("  > 0.5  reasonable rank agreement")
    print("  < 0.3  model struggles to rank this position")

    # ----------------------------------------------------------------
    # Stage 6: Generate 2026 predictions from 2025 features
    # ----------------------------------------------------------------
    # We use each player's 2025 season stats as the feature input to
    # predict how many points they will score in the 2026 season.
    _print_section("2026 predictions (from 2025 features)")

    preds_2026 = predict_for_season(
        overall.model,
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
            "feat_points",            # actual 2025 points (for reference)
            "predicted_2026_points",  # Ridge prediction for 2026
        ]
    ).sort("predicted_2026_points", descending=True)

    preds_2026.write_csv(args.out_preds)
    print(f"Wrote {preds_2026.height} rows to {args.out_preds}")
    print()
    print("Top 10 predicted players for 2026:")
    print(preds_2026.head(10))


if __name__ == "__main__":
    main()