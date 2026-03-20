"""Utilities for deriving player age from biographical roster data.

Why a separate module?
----------------------
Age derivation is a self-contained transformation that takes player
birth dates and season years as inputs and produces an age column as
output. Keeping it separate from season_features.py makes it easy to
test in isolation and reuse in other parts of the pipeline.
"""

from __future__ import annotations

from typing import List, Optional

import polars as pl


def derive_season_age(
    players: pl.DataFrame,
    *,
    player_id_col: str = "gsis_id",
    birth_date_col: str = "birth_date",
    seasons: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Derive each player's age at the start of each NFL season.

    NFL seasons start in September. We define a player's age for a
    given season as their age on September 1st of that season year.
    This gives a consistent, comparable age figure across all players
    and seasons.

    Algorithm
    ---------
    1. Parse birth_date strings into Polars Date values.
    2. For each season, compute September 1st of that year.
    3. Age = floor((season_start_date - birth_date) / 365.25).
       We use 365.25 to account for leap years.

    Parameters
    ----------
    players : pl.DataFrame
        Roster table from nflreadpy.load_players() containing at least
        gsis_id and birth_date columns.
    player_id_col : str
        Name of the player ID column in the players table.
        nflreadpy uses 'gsis_id' which matches player_id in weekly
        stats. Default: 'gsis_id'.
    birth_date_col : str
        Name of the birth date column. Default: 'birth_date'.
    seasons : List[int], optional
        Specific seasons to compute ages for. If None, computes for
        seasons 2000 through 2030 to cover all realistic training data.

    Returns
    -------
    pl.DataFrame
        One row per (player_id, season) with a derived 'age' column.
        Players with null birth date will have null age.

    Raises
    ------
    ValueError
        If player_id_col or birth_date_col are not found in players.

    Examples
    --------
    >>> players_df = pl.DataFrame({
    ...     "gsis_id": ["00-0033873"],
    ...     "birth_date": ["1995-09-17"],
    ... })
    >>> derive_season_age(players_df, seasons=[2024])
    # player_id="00-0033873", season=2024, age=28
    """
    if player_id_col not in players.columns:
        raise ValueError(
            f"Player ID column {player_id_col!r} not found. "
            f"Available columns: {players.columns}"
        )
    if birth_date_col not in players.columns:
        raise ValueError(
            f"Birth date column {birth_date_col!r} not found. "
            f"Available columns: {players.columns}"
        )

    if seasons is None:
        seasons = list(range(2000, 2031))

    # Keep only the two columns we need to avoid carrying all 39 player
    # columns through the cross join below, which would be wasteful.
    players_slim = (
        players
        .select([
            pl.col(player_id_col).alias("player_id"),
            pl.col(birth_date_col),
        ])
        # Drop players with no birth date — we cannot compute age
        # for them and keeping nulls would silently pollute the join.
        .filter(pl.col(birth_date_col).is_not_null())
    )

    # Parse birth_date to a proper Polars Date type.
    # nflreadpy stores dates as strings in "YYYY-MM-DD" format.
    # strict=False means unparseable values become null rather than
    # raising an error, which is safer for real-world data.
    players_slim = players_slim.with_columns(
        pl.col(birth_date_col)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("birth_date_parsed")
    )

    # Build a small DataFrame of seasons so we can cross join and
    # compute age for all (player, season) combinations at once.
    seasons_df = pl.DataFrame({"season": seasons})

    # Cross join: every player paired with every season.
    # This is efficient here because the players table has ~24k rows
    # and the seasons list is short (~30 entries), giving ~720k rows
    # total — well within memory for a standard machine.
    crossed = players_slim.join(seasons_df, how="cross")

    # Compute age as of September 1st of each season year.
    # We use September 1st because that is approximately when the NFL
    # regular season begins, giving a consistent reference point.
    crossed = crossed.with_columns(
        pl.date(pl.col("season"), pl.lit(9), pl.lit(1))
        .alias("season_start")
    )

    # Age in years = days between birth and season start / 365.25.
    # We floor to whole years (e.g. 28.7 years old → age 28).
    crossed = crossed.with_columns(
        (
            (pl.col("season_start") - pl.col("birth_date_parsed"))
            .dt.total_days()
            / 365.25
        )
        .floor()
        .cast(pl.Float64)
        .alias("age")
    )

    return crossed.select(["player_id", "season", "age"])


def join_age_to_features(
    features: pl.DataFrame,
    age_lookup: pl.DataFrame,
) -> pl.DataFrame:
    """Left-join derived age onto the season-level feature table.

    We use a left join so that players without a matching age entry
    are kept in the features table with a null age, rather than being
    silently dropped. The runner handles null ages by dropping those
    rows before training and reports how many were lost.

    Parameters
    ----------
    features : pl.DataFrame
        Season-level feature table with player_id and season columns.
    age_lookup : pl.DataFrame
        Output of derive_season_age — one row per (player_id, season)
        with an 'age' column.

    Returns
    -------
    pl.DataFrame
        The features table with an 'age' column appended.
    """
    # Drop any existing age column first to avoid a column name conflict
    # on the join (e.g. if a previous run already added age).
    if "age" in features.columns:
        features = features.drop("age")

    return features.join(
        age_lookup,
        on=["player_id", "season"],
        how="left",
    )