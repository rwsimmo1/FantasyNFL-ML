from __future__ import annotations

import polars as pl

from .base import WeeklyPlayerStatsSource


class NflverseWeeklyPlayerStatsSource(WeeklyPlayerStatsSource):
    """
    Adapter around nflreadpy.

    This is intentionally small so:
    - unit tests can mock the Protocol
    - switching data sources is easy
    """

    def load_weekly_player_stats(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl  # local import to keep dependency isolated

        return nfl.load_player_stats(seasons=season, summary_level="week")