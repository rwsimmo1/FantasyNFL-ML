import os
import polars as pl

from fantasy_ml.run_baseline import build_baseline_top24


class FakeWeeklySource:
    def load_weekly_player_stats(self, season: int) -> pl.DataFrame:
        assert season == 2025
        return pl.DataFrame(
            {
                "season": [2025, 2025, 2025, 2025],
                "week": [1, 2, 1, 1],
                "player_id": ["p1", "p1", "p2", "p3"],
                "player_display_name": ["A", "A", "B", "Kicker"],
                "position": ["RB", "RB", "RB", "K"],
                "rushing_yards": [100, 50, 70, 0],
                "rushing_tds": [1, 0, 0, 0],
                "receptions": [2, 1, 0, 0],
                "receiving_yards": [10, 0, 0, 0],
                "receiving_tds": [0, 0, 0, 0],
                "passing_yards": [0, 0, 0, 0],
                "passing_tds": [0, 0, 0, 0],
                "passing_interceptions": [0, 0, 0, 0],
                "sack_fumbles_lost": [0, 0, 0, 0],
                "rushing_fumbles_lost": [0, 0, 0, 0],
                "receiving_fumbles_lost": [0, 0, 0, 0],
                "fg_made": [0, 0, 0, 2],
                "pat_made": [0, 0, 0, 1],
            }
        )


def test_build_baseline_top24_works_without_network(monkeypatch):
    # Force scoring mode for the test (avoid loading actual .env)
    monkeypatch.setenv("FANTASY_SCORING", "PPR")

    out = build_baseline_top24(FakeWeeklySource(), season=2025, top_n=24)
    assert out.shape[0] >= 2
    assert set(out["position"].to_list()).issubset({"RB", "K"})