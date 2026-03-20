from __future__ import annotations

import nflreadpy
import polars as pl

from fantasy_ml.data_sources.base import WeeklyPlayerStatsSource


class NflverseWeeklyPlayerStatsSource(WeeklyPlayerStatsSource):
    """Concrete data source backed by the nflreadpy package.

    nflreadpy is a Python wrapper around the nflverse data repository
    which hosts cleaned, ready-to-use NFL statistics in parquet format.
    It returns native Polars DataFrames — no pandas conversion needed.
    """

    def load_weekly_player_stats(self, season: int) -> pl.DataFrame:
        """Load weekly player stats for a single NFL season.

        Parameters
        ----------
        season : int
            The NFL season year (e.g. 2024 for the 2024-25 season).

        Returns
        -------
        pl.DataFrame
            One row per (player, week) with all available stat columns.
        """
        # nflreadpy.load_player_stats returns a native Polars DataFrame
        # so no pandas conversion is required.
        return nflreadpy.load_player_stats(seasons=season)

    def load_players(self) -> pl.DataFrame:
        """Load the nflverse players roster table.

        This table contains one row per player and includes biographical
        data such as birth_date, which we use to derive player age.
        The players table is not season-specific — it covers all players
        historically available in nflverse.

        Returns
        -------
        pl.DataFrame
            One row per player with columns including player_id,
            display_name, position, and birth_date.
        """
        # nflreadpy.load_players returns a native Polars DataFrame.
        return nflreadpy.load_players()