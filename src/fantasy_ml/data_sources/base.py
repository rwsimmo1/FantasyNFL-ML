from __future__ import annotations

from typing import Protocol
import polars as pl


class WeeklyPlayerStatsSource(Protocol):
    """
    Interface for anything that can provide weekly player stats.

    By depending on this Protocol, our pipeline can accept:
    - nflverse/nflreadpy
    - a CSV file source
    - a database source
    - an API wrapper
    - mocked test source
    """

    def load_weekly_player_stats(self, season: int) -> pl.DataFrame: ...