# Stable State — Pass 1 + Pass 2

**Date:** March 19, 2026  
**Branch/Commit:** Current main branch (post Pass 1 + Pass 2)

---

## What was built

This document records the stable state of the FantasyNFL-ML pipeline
after completing Pass 1 and Pass 2 of the Ridge regression implementation.

---

## Pipeline overview

The full pipeline runs in six stages via `run_train_ridge.py`:

```
Weekly stats (nflverse)
    │
    ▼
Season totals (weekly_to_season.py)
    │
    ▼
Feature engineering (season_features.py)
    │
    ▼
Supervised dataset (next_season.py)
    │
    ▼
Ridge regression training + evaluation (ridge.py + evaluation.py)
    │
    ▼
2026 predictions CSV
```

---

## Files added or changed

| File | Status | What it does |
|---|---|---|
| `src/fantasy_ml/features/season_features.py` | Updated | Adds raw count features and per-game rate features |
| `src/fantasy_ml/model/ridge.py` | Updated | Core Ridge training, naive baseline, per-position training |
| `src/fantasy_ml/model/evaluation.py` | New | Top-N slice metrics, Spearman rank correlation |
| `src/fantasy_ml/run_train_ridge.py` | Updated | Full pipeline runner with all evaluation stages |
| `tests/test_model_ridge.py` | Updated | Tests for all Ridge functions |
| `tests/test_model_evaluation.py` | New | Tests for all evaluation functions |
| `tests/test_features_season_features.py` | Updated | Tests for new feature columns |
| `README.md` | Updated | Install/test/run commands + ML concept explanations |
| `docs/stable-state-pass1-pass2.md` | New | This document |

---

## Feature columns

### Raw count features (Pass 1 addition)

| Column | Source | Description |
|---|---|---|
| `feat_points` | `season_fantasy_points` | Total fantasy points for the season |
| `feat_games` | `games_in_data` | Number of games played |
| `feat_carries` | `carries_sum` | Total rushing attempts for the season |
| `feat_targets` | `targets_sum` | Total receiving targets for the season |
| `feat_pass_att` | `pass_attempts_sum` | Total pass attempts for the season |

### Per-game rate features (existing, kept)

| Column | Description |
|---|---|
| `feat_ppg` | Average fantasy points per game |
| `feat_carries_pg` | Average carries per game |
| `feat_targets_pg` | Average targets per game |
| `feat_pass_att_pg` | Average pass attempts per game |

### Features used as Ridge inputs in the runner

```python
feature_cols = [
    "feat_ppg",         # average fantasy points per game
    "feat_carries_pg",  # average carries per game   (RB usage)
    "feat_targets_pg",  # average targets per game   (WR/TE usage)
    "feat_pass_att_pg", # average pass attempts per game (QB usage)
]
```

---

## Functions added

### `src/fantasy_ml/model/ridge.py`

| Function | Description |
|---|---|
| `compare_ridge_to_naive` | Compares Ridge to naive baseline (y_pred = feat_points) on same time split |
| `train_ridge_by_position` | Trains one Ridge model per position, returns per-position + overall metrics |
| `_compute_metrics` | Shared helper that computes MAE, RMSE, R² from two arrays |

### `src/fantasy_ml/model/evaluation.py`

| Function | Description |
|---|---|
| `top_n_per_position_metrics` | MAE/RMSE/R² on top-N actual scorers per position |
| `top_n_overall_metrics` | MAE/RMSE/R² on top-N actual scorers overall |
| `spearman_rank_by_position` | Spearman rank correlation between predicted and actual within each position |

---

## Test results (all passing)

```
pytest tests/test_model_ridge.py
tests/test_model_evaluation.py
tests/test_features_season_features.py -v

19 passed in ~3.5s
```

| Test file | Tests | Result |
|---|---|---|
| `test_model_ridge.py` | 9 | ✅ All passed |
| `test_model_evaluation.py` | 10 | ✅ All passed |
| `test_features_season_features.py` | 2 | ✅ All passed |

---

## Runner output (March 19, 2026)

Run with default parameters:
```
python -m fantasy_ml.run_train_ridge
Train seasons: [2016..2022]   Test seasons: [2023, 2024]
```

### Overall Ridge
| Metric | Value |
|---|---|
| MAE | 47.77 |
| RMSE | 67.74 |
| R² | 0.484 |

### Ridge vs Naive baseline
| Metric | Ridge | Naive | Winner |
|---|---|---|---|
| MAE | 47.77 | 47.32 | Naive (by 0.45 pts) |
| RMSE | 67.74 | 69.95 | Ridge |
| R² | 0.48 | 0.45 | Ridge |

### Per-position Ridge
| Position | MAE | RMSE | R² |
|---|---|---|---|
| K | 35.00 | 42.02 | 0.034 |
| QB | 83.94 | 110.44 | 0.263 |
| RB | 50.89 | 72.77 | 0.528 |
| TE | 31.14 | 44.31 | 0.521 |
| WR | 44.80 | 59.92 | 0.535 |
| **Overall** | **47.78** | **68.15** | **0.478** |

### Top-24 per position slice metrics
| Position | MAE | RMSE | R² |
|---|---|---|---|
| K | 47.60 | 51.47 | -6.52 |
| QB | 127.72 | 147.68 | -5.54 |
| RB | 146.28 | 157.98 | -6.49 |
| TE | 78.40 | 85.82 | -4.49 |
| WR | 95.81 | 113.38 | -2.74 |

> **Note:** Negative R² on top-24 slices means the model performs worse
> than predicting the mean for elite players. This is the primary
> weakness to address in the next pass.

### Top-100 overall slice metrics
| n | MAE | RMSE | R² |
|---|---|---|---|
| 100 | 110.84 | 130.27 | -3.22 |

### Spearman rank correlation by position
| Position | n | Spearman | Verdict |
|---|---|---|---|
| K | 66 | 0.299 | Borderline |
| QB | 128 | 0.545 | Reasonable |
| RB | 214 | 0.769 | **Strong** |
| TE | 205 | 0.730 | **Strong** |
| WR | 348 | 0.749 | **Strong** |

### Top 10 predicted players for 2026
| Player | Position | 2025 actual | 2026 predicted |
|---|---|---|---|
| Puka Nacua | WR | 452.6 | 296.5 |
| Christian McCaffrey | RB | 458.4 | 284.7 |
| Josh Allen | QB | 414.8 | 276.8 |
| Jahmyr Gibbs | RB | 366.9 | 257.3 |
| Bijan Robinson | RB | 368.8 | 255.8 |
| Jaxon Smith-Njigba | WR | 408.8 | 255.4 |
| Matthew Stafford | QB | 411.3 | 249.9 |
| Jonathan Taylor | RB | 360.3 | 248.7 |
| Ja'Marr Chase | WR | 313.6 | 244.0 |
| Patrick Mahomes | QB | 281.7 | 241.4 |

> **Note:** Predicted totals are noticeably lower than 2025 actuals.
> This is regression to the mean — a known behaviour of linear models
> when elite players score well above average.

---

## Known weaknesses to address in next pass

| Weakness | Evidence | Likely fix |
|---|---|---|
| Ridge barely beats naive on MAE | MAE: Ridge 47.77 vs Naive 47.32 | Add richer features (age, injury, role) |
| Negative R² on top-24 slices | All positions negative | Model undertrained on elite players |
| QB R² low (0.26) | Per-position table | Add QB-specific features (e.g. receiver quality) |
| K nearly unpredictable (R² 0.03) | Per-position table | Consider excluding K or separate modelling approach |
| Regression to mean in predictions | Top 10 predicted vs actual gap | Consider quantile regression or ensemble |

---

## How to run

```powershell
# Install
python -m pip install -e .

# Run all tests
pytest -q

# Run the full pipeline with defaults
python -m fantasy_ml.run_train_ridge

# Run with custom parameters
python -m fantasy_ml.run_train_ridge \
    --start-season 2016 \
    --end-season 2025 \
    --train-end-season 2022 \
    --alpha 1.0 \
    --out-preds pred_2026_from_2025.csv
```