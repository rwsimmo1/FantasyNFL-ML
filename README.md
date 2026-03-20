# FantasyNFL-ML

Learning project: build a Python ML pipeline that uses NFL player statistics (via `nflreadpy`/nflverse) to **predict 2026 regular-season fantasy points** from **2025 stats**, then output **top 24** players per position:

- QB / RB / WR / TE / K / DST

## Quick start

### 1) Create & activate a virtual environment

```bash
python -m venv .venv
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 3) Configure scoring via `.env`

Copy the example env file:

```bash
cp .env.example .env
```

Then edit `.env` and set:

- `FANTASY_SCORING=PPR` or `HALF_PPR` or `STANDARD`
- `SCORING_PROVIDER=ESPN` or `YAHOO` (defaults to `ESPN`)

Notes:
- Passing TDs are **6 points** (`pass_td=6`)
- K and DST are **MVP scoring for now** (simple scoring; we’ll refine later)

### 4) Run unit tests

```bash
pytest -q
```

### 5) Run (placeholder)

Once the first runnable pipeline script is added, you’ll run something like:

```bash
python -m fantasy_ml.run_baseline --season 2025 --top-n 24
```

(We’ll add this after the scoring + ranking modules and tests are in place.)

## Learning plan (short checkpoints)

As we implement each stage, we’ll map it to a short learning checkpoint:

- StatQuest (YouTube): Linear Regression, Ridge/Lasso, Bias-Variance
- StatQuest: Decision Trees, Random Forests, Gradient Boosting
- Book: *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Géron)
- scikit-learn docs: model evaluation, cross validation, and data leakage pitfalls

## Project principles

- Every function should be written in a **testable** way.
- Every function should have an accompanying **unit test**.
- Network calls (downloading nflverse data) should be wrapped so unit tests can **mock** them.
- Time-based splits for training/validation/testing to avoid **data leakage**.

## How to read MAE, RMSE, and R²

When evaluating the Ridge regression model, these three metrics are the most
important:

- **MAE (Mean Absolute Error)**  
  Average size of the prediction error.  
  If MAE = `12`, predictions are off by about **12 points** on average.  
  **Lower is better.**

- **RMSE (Root Mean Squared Error)**  
  Similar to MAE, but gives extra weight to large mistakes.  
  If RMSE is much higher than MAE, the model makes some big misses.  
  **Lower is better.**

- **R² (R-squared)**  
  Measures how much of the target variation the model explains.  
  - `1.0` = perfect fit  
  - `0.0` = no better than predicting the average  
  - `< 0.0` = worse than predicting the average  
  **Higher is better.**

### Quick rule of thumb

- Start by tracking **MAE** for "typical error size."
- Use **RMSE** to check whether big errors are a problem.
- Use **R²** as an overall quality score for explained variance.

---

## Install

```powershell
cd d:\development\FantasyNFL-ML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest polars scikit-learn nflreadpy
```

## Run all tests

```powershell
pytest -q
```

## Run specific test files

```powershell
pytest tests\test_model_ridge.py tests\test_model_evaluation.py tests\test_features_season_features.py -v
```

## Run ridge training runner

```powershell
python -m fantasy_ml.runners.run_train_ridge
```

---

## Understanding the evaluation pipeline

Once the model is trained, we run several evaluation steps to understand
**how well it works** and **where it adds value**. This section explains
each step in plain language.

---

### Step 1 — Compare Ridge to a naive baseline

**What it does**
Runs the same time-aware split and compares two sets of predictions
side by side:

| Model | Prediction rule |
|---|---|
| Naive baseline | Next season = this season's points (no model needed) |
| Ridge regression | Next season = weighted combination of feature columns |

**Why it matters**
This is one of the most important habits in ML. If a sophisticated model
cannot beat a trivially simple rule, the model is not adding value. Always
establish a baseline before claiming a model is useful.

**What to look for**
Ridge MAE should be lower than naive MAE. If it is not, the features or
regularization strength may need adjustment.

---

### Step 2 — Train one Ridge model per position

**What it does**
Trains five separate Ridge models — one each for QB, RB, WR, TE, and K —
then reports per-position metrics and combined overall metrics.

**Why it matters**
Players at different positions score fantasy points in completely different
ways:
- QBs score through passing touchdowns and yards
- RBs score through rushing yards and receptions
- Ks score through field goals and extra points

A single model trained on all positions at once must compromise between all
of these patterns. Per-position models can specialise on each position's
unique scoring behaviour.

**What to look for**
Compare per-position MAE to the combined model MAE. Some positions
(e.g. QB) tend to be more predictable than others (e.g. RB).

---

### Step 3 — Top-N slice metrics

**What it does**
Instead of measuring accuracy across all players (including bench-warmers
and injured players who score near zero), slice metrics evaluate the model
only on the players that matter most in fantasy:

| Slice | Who is included |
|---|---|
| Top-24 per position | The 24 highest-scoring actual players at each position |
| Top-100 overall | The 100 highest-scoring actual players across all positions |

**Why it matters**
A model that is accurate overall but terrible on elite players is not useful
for fantasy drafts. Slice metrics reveal how well the model performs
**where it actually matters**.

**What to look for**
Top-N MAE will usually be higher than overall MAE because elite players
have more variance in their scores. The goal is to minimise it.

---

### Step 4 — Spearman rank correlation by position

**What it does**
Instead of comparing raw point totals, Spearman correlation ranks each
player within their position (1st, 2nd, 3rd, etc.) by actual points and
by predicted points, then measures how well those orderings agree.

| Score | Meaning |
|---|---|
| 1.0 | Model ranks all players in perfect order |
| 0.0 | Model rankings are random (no relationship to actual) |
| -1.0 | Model ranks players in completely reversed order |

**Why it matters**
In fantasy football, **ranking players correctly is often more important
than predicting exact point totals**. A model that says Player A will
score 180 points and Player B will score 160 points is useful even if the
real scores turn out to be 220 and 195 — as long as the ordering is right.
Spearman correlation captures this ranking quality directly.

**What to look for**
A Spearman correlation above **0.5** within a position is a reasonable
starting target. Above **0.7** is strong. Below **0.3** suggests the model
is struggling to rank that position reliably.

---

### Learning references

**Baseline models and why they matter**
- [Scikit-learn dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)

**Regression metrics (MAE, RMSE, R²)**
- [Scikit-learn regression metrics guide](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [StatQuest: R-squared explained (YouTube)](https://www.youtube.com/watch?v=2AQKmw14mHM)

**Ridge regression and regularization**
- [Scikit-learn Ridge regression user guide](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
- [StatQuest: Regularization Part 2 - Ridge (YouTube)](https://www.youtube.com/watch?v=Q81RR3yKn30)

**Spearman rank correlation**
- [Wikipedia: Spearman's rank correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
- [SciPy spearmanr documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)

**Slice-based evaluation**
- [Google PAIR: What-If Tool (slice evaluation)](https://pair-code.github.io/what-if-tool/)

## Feature notes

### `feat_age`
`feat_age` is the player's age (in years) for that season, derived from nflverse player metadata (`birth_date`).

- **How it is computed:** age on **September 1** of the season year (used as a consistent season-start reference date).
- **Why it exists:** player performance is age-dependent (development, peak years, and decline), so age helps the model separate players with similar recent stats but different career stages.
- **How it is used in modeling:** `feat_age` is included as an input feature for Ridge regression.  
  The pipeline also uses `feat_age_sq` (age squared) so the model can learn a non-linear aging curve.

In the 2026 prediction output (`pred_2026_from_2025.csv`), `feat_age` is shown as a reference column so users can interpret predictions in age context.