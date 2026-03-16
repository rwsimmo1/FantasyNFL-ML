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

- Start by tracking **MAE** for “typical error size.”
- Use **RMSE** to check whether big errors are a problem.
- Use **R²** as an overall quality score for explained variance.