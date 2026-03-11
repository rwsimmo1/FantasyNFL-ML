import polars as pl

from fantasy_ml.scoring import FantasyScoring, add_fantasy_points, scoring_from_mode
from fantasy_ml.scoring_rules import get_scoring_rules

def test_scoring_from_mode_reception_values():
    assert scoring_from_mode("PPR").rec == get_scoring_rules("PPR", "ESPN")["rec"]
    assert scoring_from_mode("HALF_PPR").rec == get_scoring_rules("HALF_PPR", "ESPN")["rec"]
    assert scoring_from_mode("STANDARD").rec == get_scoring_rules("STANDARD", "ESPN")["rec"]


def test_scoring_from_mode_supports_provider_selection():
    espn = scoring_from_mode("PPR", provider="ESPN")
    yahoo = scoring_from_mode("PPR", provider="YAHOO")
    assert espn == FantasyScoring(**get_scoring_rules("PPR", "ESPN"))
    assert yahoo == FantasyScoring(**get_scoring_rules("PPR", "YAHOO"))


def test_add_fantasy_points_qb_like_row_pass_td_6():
    df = pl.DataFrame(
        {
            "passing_yards": [4000],
            "passing_tds": [30],
            "interceptions": [10],
            "rushing_yards": [200],
            "rushing_tds": [2],
            "receptions": [0],
            "receiving_yards": [0],
            "receiving_tds": [0],
            "fumbles_lost": [3],
        }
    )

    scoring = FantasyScoring(rec=1.0, pass_td=6.0)
    out = add_fantasy_points(df, scoring)
    pts = out["fantasy_points"][0]

    expected = (
        4000 * scoring.pass_yd
        + 30 * scoring.pass_td
        + 10 * scoring.pass_int
        + 200 * scoring.rush_yd
        + 2 * scoring.rush_td
        + 0 * scoring.rec
        + 0 * scoring.rec_yd
        + 0 * scoring.rec_td
        + 3 * scoring.fumble_lost
    )
    assert pts == expected


def test_add_fantasy_points_missing_columns_treated_as_zero():
    # RB-ish row but only a subset of columns provided
    df = pl.DataFrame({"rushing_yards": [100], "rushing_tds": [1]})
    scoring = FantasyScoring(rec=1.0)
    out = add_fantasy_points(df, scoring)
    assert "fantasy_points" in out.columns
    assert out["fantasy_points"][0] == 100 * scoring.rush_yd + 1 * scoring.rush_td


def test_add_fantasy_points_dst_mvp_columns():
    df = pl.DataFrame(
        {
            "def_sacks": [4],
            "def_interceptions": [2],
            "def_fumble_recoveries": [1],
            "def_tds": [1],
        }
    )
    scoring = FantasyScoring(rec=1.0)
    out = add_fantasy_points(df, scoring)
    pts = out["fantasy_points"][0]
    expected = (
        4 * scoring.def_sack
        + 2 * scoring.def_int
        + 1 * scoring.def_fumble_recovery
        + 1 * scoring.def_td
    )
    assert pts == expected