import polars as pl

from fantasy_ml.scoring import FantasyScoring
from fantasy_ml.scoring_nflverse import add_fantasy_points_nflverse_weekly


def test_add_fantasy_points_nflverse_weekly_basic_qb():
    df = pl.DataFrame(
        {
            "passing_yards": [300.0],
            "passing_tds": [2.0],
            "passing_interceptions": [1.0],
            "rushing_yards": [10.0],
            "rushing_tds": [0.0],
            "receptions": [0.0],
            "receiving_yards": [0.0],
            "receiving_tds": [0.0],
            "sack_fumbles_lost": [0.0],
            "rushing_fumbles_lost": [0.0],
            "receiving_fumbles_lost": [0.0],
        }
    )

    scoring = FantasyScoring(pass_td=6.0, rec=1.0)
    out = add_fantasy_points_nflverse_weekly(df, scoring, out_col="fp")
    fp = out["fp"][0]

    expected = 300 * (1 / 25) + 2 * 6 + 1 * (-2) + 10 * (1 / 10)
    assert fp == expected


def test_add_fantasy_points_nflverse_weekly_includes_kicking_and_fumbles_lost_sum():
    df = pl.DataFrame(
        {
            "fg_made": [3],
            "pat_made": [2],
            "rushing_fumbles_lost": [1],
            "receiving_fumbles_lost": [0],
            "sack_fumbles_lost": [1],
        }
    )
    scoring = FantasyScoring(pass_td=6.0, rec=1.0)
    out = add_fantasy_points_nflverse_weekly(df, scoring, out_col="fp")
    fp = out["fp"][0]

    expected = 3 * 3.0 + 2 * 1.0 + (1 + 0 + 1) * (-2.0)
    assert fp == expected


def test_add_fantasy_points_nflverse_weekly_missing_columns_are_zero():
    df = pl.DataFrame({"rushing_yards": [100], "rushing_tds": [1]})
    out = add_fantasy_points_nflverse_weekly(df, FantasyScoring(rec=1.0), out_col="fp")
    assert out["fp"][0] == 100 * (1 / 10) + 1 * 6