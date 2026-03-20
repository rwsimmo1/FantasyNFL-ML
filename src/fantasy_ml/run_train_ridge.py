"""Entry-point script for training a Ridge regression model on NFL fantasy data.

This script ties together the full data pipeline in six stages:

1. Load weekly player stats for a range of seasons (e.g. 2016-2025).
2. Score each weekly row and aggregate to season totals.
3. Build engineered feature columns from the season totals.
4. Derive player age from the nflverse players roster table and join
   it onto the feature table.
5. Build a supervised dataset where each row is:
       (player, season t) -> target = fantasy points in season t+1
6. Train Ridge regression using a time-aware split, then evaluate:
       - Overall Ridge metrics (MAE, RMSE, R²)
       - Ridge vs naive baseline comparison
       - Per-position Ridge models
       - Top-N slice metrics (top-24 per position, top-100 overall)
       - Spearman rank correlation by position
7. Generate 2026 predictions from each player's 2025 season stats.

How to run:
    python -m fantasy_ml.run_train_ridge
    python -m fantasy_ml.run_train_ridge --start-season 2018 --end-season 2025

See argparse arguments below for all available options.
"""

from __future__ import annotations

import argparse
from typing import List

import polars as pl

from fantasy_ml.config import load_config
from fantasy_ml.data_sources.base import WeeklyPlayerStatsSource
from fantasy_ml.data_sources.nflverse_source import (
    NflverseWeeklyPlayerStatsSource,
)
from fantasy_ml.datasets.next_season import build_next_season_dataset
from fantasy_ml.features.age_features import (
    derive_season_age,
    join_age_to_features,
)
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
    source: NflverseWeeklyPlayerStatsSource,
    *,
    seasons: List[int],
) -> pl.DataFrame:
    """Load weekly stats for each season and build the ML feature table.

    This function isolates data loading and feature engineering so the
    rest of main() can focus on model training. Keeping IO (input/output)
    separate from computation is a good software design habit — it makes
    each piece easier to test and swap out independently.

    Parameters
    ----------
    source : NflverseWeeklyPlayerStatsSource
        A data-source object that knows how to fetch weekly player stats
        and the players roster table.
    seasons : List[int]
        List of NFL seasons to load (e.g. [2016, 2017, ..., 2025]).

    Returns
    -------
    pl.DataFrame
        One row per (player, season) with all feat_* columns attached,
        including feat_age and feat_age_sq derived from birth_date.
    """
    # Load the scoring weights (PPR, half-PPR, standard) from project config.
    # Scoring mode is set in the .env file so it can be changed without
    # touching any Python code.
    cfg = load_config()
    scoring = scoring_from_mode(cfg.fantasy_scoring)

    frames: List[pl.DataFrame] = []

    for s in seasons:
        # --- Step 1: Load raw weekly stats for one season ---
        # Each row in `weekly` represents one player's stats for one game.
        weekly = source.load_weekly_player_stats(s)

        # Keep only the five fantasy-relevant positions.
        # Positions like FB, OL, or DL do not score fantasy points so
        # including them would add noise to the model.
        weekly = weekly.filter(
            pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
        )

        # --- Step 2: Score each week ---
        # add_fantasy_points_nflverse_weekly applies the scoring weights
        # (e.g. 1 pt per reception in PPR, 6 pts per passing TD) to
        # produce a single fantasy_points_calc column per row.
        weekly_scored = add_fantasy_points_nflverse_weekly(
            weekly,
            scoring,
            out_col="fantasy_points_calc",
        )

        # --- Step 3: Aggregate weekly rows into season totals ---
        # One game = one weekly row, so this collapses ~17-18 rows per
        # player into a single season-total row (sum of all games).
        season_totals = season_totals_from_weekly(
            weekly_scored,
            points_col="fantasy_points_calc",
            out_points_col="season_fantasy_points",
        )

        frames.append(season_totals)

    # Stack all seasons into one DataFrame.
    # diagonal_relaxed allows columns that exist in some seasons but not
    # others (e.g. a new stat column added in a later nflverse release).
    all_seasons = pl.concat(frames, how="diagonal")

    # --- Step 4: Engineer features ---
    # build_season_features converts raw season totals into ML-ready
    # feature columns: feat_ppg, feat_carries_pg, etc.
    # These engineered columns become the X matrix for Ridge regression.
    features = build_season_features(
        all_seasons,
        points_col="season_fantasy_points",
    )

    # --- Step 5: Derive and join player age ---
    # The nflverse weekly stats do not include an age column, but the
    # players roster table has birth_date. We compute each player's age
    # on September 1st of each season year (approximately when the NFL
    # season starts) and join it onto the feature table.
    #
    # Why age matters:
    # Players typically improve through their mid-to-late 20s, peak around
    # age 26-29, then decline. Without age, the model cannot distinguish
    # a 24-year-old on the rise from a 34-year-old in decline, even if
    # both scored the same points last season.
    print("Loading player roster for age derivation...")
    players = source.load_players()

    age_lookup = derive_season_age(
        players,
        seasons=list(range(
            min(seasons) - 1,  # include one extra year for safety
            max(seasons) + 2,
        )),
    )

    features = join_age_to_features(features, age_lookup)

    # Report age coverage so the user can spot data issues.
    null_age = features.filter(pl.col("age").is_null()).height
    total = features.height
    print(
        f"Age coverage: {total - null_age}/{total} players have age data "
        f"({null_age} null)."
    )

    # build_season_features needs the age column present BEFORE it runs
    # so that feat_age and feat_age_sq are computed correctly.
    # We rebuild the features now that age is joined in.
    features = build_season_features(
        # Pass the raw season totals with age attached.
        # We reconstruct by joining age back onto all_seasons.
        all_seasons.join(
            age_lookup,
            on=["player_id", "season"],
            how="left",
        ),
        points_col="season_fantasy_points",
    )

    return features


def _print_section(title: str) -> None:
    """Print a visible section header to make console output easy to scan.

    Parameters
    ----------
    title : str
        The section title to display.
    """
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    """Parse CLI arguments, run the full pipeline, and write predictions.

    Pipeline overview
    -----------------
    1. Load + feature-engineer all seasons via load_and_build_features.
    2. Build the supervised (X, y) dataset — pairs season-t features
       with season-(t+1) target points.
    3. Train overall Ridge and print holdout metrics.
    4. Compare Ridge to naive baseline on the same split.
    5. Train per-position Ridge models and print per-position metrics.
    6. Print top-N slice metrics and Spearman rank correlations.
    7. Generate 2026 predictions from 2025 features and write to CSV.
    """
    # ----------------------------------------------------------------
    # CLI argument definitions
    # argparse lets users change settings from the command line without
    # editing the source code. Each add_argument call defines one option.
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Train Ridge and generate next-season fantasy predictions."
        )
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
    # Example: train_end=2022 means 2016-2022 train, 2023-2025 test.
    # Using a time-aware split (rather than random) prevents the model
    # from "seeing the future" during training, which would inflate metrics.
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
    # Regularization prevents overfitting by penalising large coefficients.
    # Higher alpha = stronger penalty = simpler model (may underfit).
    # Lower alpha  = weaker penalty  = behaves like ordinary least squares.
    # alpha=1.0 is a safe starting value; tune it after reviewing results.
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
        help=(
            "Output CSV path for 2026 predictions "
            "(default: pred_2026_from_2025.csv)."
        ),
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Stage 1-4: Load data, build features, and join age
    # ----------------------------------------------------------------
    seasons = list(range(args.start_season, args.end_season + 1))

    # NflverseWeeklyPlayerStatsSource is the concrete implementation that
    # fetches data from the nflverse repository. Swapping this object for
    # a different source (e.g. a CSV-backed source for offline testing)
    # requires no other code changes in this file.
    source = NflverseWeeklyPlayerStatsSource()
    features = load_and_build_features(source, seasons=seasons)

    # ----------------------------------------------------------------
    # Stage 5: Build the supervised (X, y) dataset
    # ----------------------------------------------------------------
    # build_next_season_dataset joins each player's season-t features to
    # their season-(t+1) points total. Only players present in BOTH
    # consecutive seasons are kept (inner join).
    #
    # The result is a "supervised" dataset — each row has:
    #   X = feature values from season t  (what we know now)
    #   y = fantasy points in season t+1  (what we want to predict)
    supervised = build_next_season_dataset(
        features,
        target_col="feat_points",
        out_target_col="target_next_season_points",
    )

    # These are the input columns Ridge uses to make predictions (X matrix).
    # Per-game rates are preferred over raw totals so that players who
    # missed games are not unfairly penalised for lower raw counts.
    #
    # Age features (Pass 3 addition):
    # feat_age lets the model learn that younger players tend to improve
    # and older players tend to decline. feat_age_sq (age squared) captures
    # the non-linear (curved) shape of this relationship — a straight line
    # through age vs performance cannot model the peak-and-decline pattern
    # of real NFL careers.
    feature_cols: List[str] = [
        "feat_ppg",           # average fantasy points per game
        "feat_carries_pg",    # average carries per game   (RB usage)
        "feat_targets_pg",    # average targets per game   (WR/TE usage)
        "feat_pass_att_pg",   # average pass attempts per game (QB usage)
        "feat_age",           # player age (older players tend to decline)
        "feat_age_sq",        # age squared (captures non-linear aging curve)
    ]

    # Ridge regression cannot handle null (missing) values in its input
    # matrix. We drop rows where any feature is null before training.
    # This mainly removes players whose birth_date is missing in nflverse.
    supervised_clean = supervised.drop_nulls(subset=feature_cols)

    # Report how many rows were dropped so the user can spot data issues.
    dropped = supervised.height - supervised_clean.height
    if dropped > 0:
        print(
            f"Note: dropped {dropped} rows with null feature values "
            f"(out of {supervised.height} total)."
        )

    # ----------------------------------------------------------------
    # Stage 6a: Overall Ridge — train and print holdout metrics
    # ----------------------------------------------------------------
    _print_section("Overall Ridge — holdout metrics")

    # train_ridge performs the time-aware split internally and returns
    # evaluation metrics on the holdout (test) seasons only. We never
    # report training-set metrics because a model can always fit its own
    # training data well — the test set tells us how it generalises.
    overall = train_ridge(
        supervised_clean,
        feature_cols=feature_cols,
        target_col="target_next_season_points",
        train_end_season=args.train_end_season,
        alpha=args.alpha,
    )

    print(f"Train seasons : {overall.train_seasons}")
    print(f"Test seasons  : {overall.test_seasons}")
    # MAE: average absolute error in fantasy points (lower is better).
    # e.g. MAE=47 means the model is off by about 47 points on average.
    print(f"MAE           : {overall.mae:.2f}")
    # RMSE: root mean squared error — penalises large errors more than MAE.
    # If RMSE is much higher than MAE, the model makes some big misses.
    print(f"RMSE          : {overall.rmse:.2f}")
    # R²: fraction of variance explained (1.0 = perfect, 0.0 = predicting
    # the mean every time, negative = worse than predicting the mean).
    print(f"R²            : {overall.r2:.4f}")

    # ----------------------------------------------------------------
    # Stage 6b: Ridge vs naive baseline
    # ----------------------------------------------------------------
    # The naive baseline predicts "next season = this season's points".
    # This is the simplest possible rule — no model required.
    # If our Ridge model cannot beat this baseline it is not adding value.
    # This is one of the most important habits in ML: always compare
    # your model to a simple rule before claiming it is useful.
    _print_section("Ridge vs Naive baseline")

    vs_naive = compare_ridge_to_naive(
        supervised_clean,
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
        print(
            f"{metric.upper():<10} {ridge_val:>10.2f} "
            f"{naive_val:>10.2f} {winner:>10}"
        )

    # ----------------------------------------------------------------
    # Stage 6c: Per-position Ridge models
    # ----------------------------------------------------------------
    # One model per position so each can specialise on that position's
    # unique scoring patterns (QB passing vs RB rushing vs K kicking).
    # A single model trained on all positions must compromise between
    # all of those patterns, which can hurt accuracy for each individual
    # position.
    _print_section("Per-position Ridge models")

    by_pos = train_ridge_by_position(
        supervised_clean,
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
    # Overall metrics are computed from concatenated per-position
    # predictions so they reflect all positions on the same test seasons.
    om = by_pos.overall_metrics
    print(f"{'Overall':<12} {om.mae:>8.2f} {om.rmse:>8.2f} {om.r2:>8.4f}")

    # ----------------------------------------------------------------
    # Stage 6d: Top-N slice metrics
    # ----------------------------------------------------------------
    # Evaluate only on the players that matter most in fantasy drafts.
    # Overall metrics include bench-warmers who score near zero; those
    # easy-to-predict low scorers can make the model look better than it
    # really is for the players that actually get drafted.
    _print_section("Top-24 per position — slice metrics")

    top24 = top_n_per_position_metrics(
        by_pos.predictions,
        n=24,
        actual_col="y_true",
        pred_col="y_pred",
    )
    print(top24)

    _print_section("Top-100 overall — slice metrics")

    # Top-100 overall gives an aggregate quality metric for the highest-
    # value players across all positions — the players most likely to be
    # drafted in a typical fantasy league.
    top100 = top_n_overall_metrics(
        by_pos.predictions,
        n=100,
        actual_col="y_true",
        pred_col="y_pred",
    )
    print(top100)

    # ----------------------------------------------------------------
    # Stage 6e: Spearman rank correlation by position
    # ----------------------------------------------------------------
    # Spearman measures whether the model ranks players in the right order
    # within each position, regardless of whether the raw point totals are
    # accurate. In fantasy football, ranking players correctly (so you
    # draft the best available player) is often more important than
    # predicting exact point totals.
    #
    # Score guide:
    #   > 0.7  strong rank agreement
    #   > 0.5  reasonable rank agreement
    #   < 0.3  model struggles to rank this position reliably
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
    # Stage 7: Generate 2026 predictions from 2025 features
    # ----------------------------------------------------------------
    # We use each player's 2025 season stats as the feature input to
    # predict how many points they will score in the 2026 season.
    # Note: season=2025 here refers to the feature row season (the input),
    # not the prediction target season (2026).
    #
    # Drop nulls from the features table so predict_for_season receives
    # a clean matrix with no null age values.
    features_clean = features.drop_nulls(subset=feature_cols)

    _print_section("2026 predictions (from 2025 features)")

    preds_2026 = predict_for_season(
        overall.model,
        features_clean,
        feature_cols=feature_cols,
        season=2025,
        out_col="predicted_2026_points",
    ).select(
        [
            "season",
            "player_id",
            "player_display_name",
            "position",
            "feat_age",               # age during 2025 season (for reference)
            "feat_points",            # actual 2025 points (for reference)
            "predicted_2026_points",  # Ridge prediction for 2026
        ]
    ).sort("predicted_2026_points", descending=True)

    # Write predictions to disk so they can be reviewed in Excel,
    # a notebook, or any downstream tool.
    preds_2026.write_csv(args.out_preds)
    print(f"Wrote {preds_2026.height} rows to {args.out_preds}")
    print()
    print("Top 10 predicted players for 2026:")
    print(preds_2026.head(10))


if __name__ == "__main__":
    main()