"""Unit tests for age_features.py.

These tests verify that player age is correctly derived from birth_date
and that the join onto the feature table behaves safely under edge cases
such as missing birth dates and missing players.
"""

from __future__ import annotations

import polars as pl
import pytest

from fantasy_ml.features.age_features import (
    derive_season_age,
    join_age_to_features,
)


def _make_players(
    gsis_ids: list,
    birth_dates: list,
) -> pl.DataFrame:
    """Build a minimal players DataFrame for testing."""
    return pl.DataFrame({
        "gsis_id": gsis_ids,
        "birth_date": birth_dates,
    })


class TestDeriveSeasonAge:
    """Tests for derive_season_age."""

    def test_correct_age_before_birthday(self) -> None:
        """Player born after Sept 1 should not yet have had birthday."""
        # Born 1995-09-17 → on 2024-09-01 they are still 28 (turn 29
        # on Sept 17).
        players = _make_players(["p1"], ["1995-09-17"])
        result = derive_season_age(players, seasons=[2024])
        assert result["age"][0] == 28.0

    def test_correct_age_on_birthday(self) -> None:
        """Player born exactly on Sept 1 turns a year older that day."""
        # Born 1995-09-01 → on 2024-09-01 they turn exactly 29.
        players = _make_players(["p1"], ["1995-09-01"])
        result = derive_season_age(players, seasons=[2024])
        assert result["age"][0] == 29.0

    def test_correct_age_after_birthday(self) -> None:
        """Player born before Sept 1 has already had their birthday."""
        # Born 1995-01-15 → on 2024-09-01 they are 29.
        players = _make_players(["p1"], ["1995-01-15"])
        result = derive_season_age(players, seasons=[2024])
        assert result["age"][0] == 29.0

    def test_multiple_seasons_produces_correct_row_count(self) -> None:
        """One player across three seasons should produce three rows."""
        players = _make_players(["p1"], ["1995-01-15"])
        result = derive_season_age(players, seasons=[2022, 2023, 2024])
        assert result.height == 3

    def test_multiple_players_and_seasons(self) -> None:
        """Two players × two seasons = four rows."""
        players = _make_players(
            ["p1", "p2"],
            ["1995-01-15", "2000-06-20"],
        )
        result = derive_season_age(players, seasons=[2023, 2024])
        assert result.height == 4

    def test_null_birth_date_is_excluded(self) -> None:
        """Players with null birth_date should be excluded from output."""
        players = _make_players(["p1", "p2"], [None, "1995-01-15"])
        result = derive_season_age(players, seasons=[2024])
        # Only p2 should appear — p1 has no birth date.
        assert result.height == 1
        assert result["player_id"][0] == "p2"

    def test_missing_player_id_col_raises(self) -> None:
        """ValueError raised when player_id_col is not in DataFrame."""
        players = pl.DataFrame({"birth_date": ["1995-01-15"]})
        with pytest.raises(ValueError, match="gsis_id"):
            derive_season_age(players, seasons=[2024])

    def test_missing_birth_date_col_raises(self) -> None:
        """ValueError raised when birth_date_col is not in DataFrame."""
        players = pl.DataFrame({"gsis_id": ["p1"]})
        with pytest.raises(ValueError, match="birth_date"):
            derive_season_age(players, seasons=[2024])

    def test_output_columns(self) -> None:
        """Output must contain exactly player_id, season, and age."""
        players = _make_players(["p1"], ["1995-01-15"])
        result = derive_season_age(players, seasons=[2024])
        assert set(result.columns) == {"player_id", "season", "age"}


class TestJoinAgeToFeatures:
    """Tests for join_age_to_features."""

    def test_age_joined_correctly(self) -> None:
        """Age value from lookup should appear on the matching row."""
        features = pl.DataFrame({
            "player_id": ["p1"],
            "season": [2024],
            "feat_ppg": [15.0],
        })
        age_lookup = pl.DataFrame({
            "player_id": ["p1"],
            "season": [2024],
            "age": [28.0],
        })
        result = join_age_to_features(features, age_lookup)
        assert result["age"][0] == 28.0

    def test_missing_player_gets_null_age(self) -> None:
        """Players not in age_lookup should have null age (left join)."""
        features = pl.DataFrame({
            "player_id": ["p1", "p2"],
            "season": [2024, 2024],
            "feat_ppg": [15.0, 10.0],
        })
        # Only p1 is in the lookup — p2 should get null age.
        age_lookup = pl.DataFrame({
            "player_id": ["p1"],
            "season": [2024],
            "age": [28.0],
        })
        result = join_age_to_features(features, age_lookup)
        assert result.height == 2
        p2_age = result.filter(pl.col("player_id") == "p2")["age"][0]
        assert p2_age is None

    def test_existing_age_column_is_replaced(self) -> None:
        """If features already has an age column it should be replaced."""
        features = pl.DataFrame({
            "player_id": ["p1"],
            "season": [2024],
            "feat_ppg": [15.0],
            # Stale age value from a previous run.
            "age": [99.0],
        })
        age_lookup = pl.DataFrame({
            "player_id": ["p1"],
            "season": [2024],
            "age": [28.0],
        })
        result = join_age_to_features(features, age_lookup)
        # Should use the lookup value, not the stale 99.0.
        assert result["age"][0] == 28.0