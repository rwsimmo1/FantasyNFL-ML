import polars as pl
import pytest

from fantasy_ml.ranking import top_n_by_position


def test_top_n_by_position_filters_to_known_positions_and_ranks():
    df = pl.DataFrame(
        {
            "player": ["a", "b", "c", "d", "e"],
            "position": ["QB", "QB", "RB", "RB", "LS"],  # LS should be filtered out
            "fantasy_points": [10.0, 20.0, 5.0, 15.0, 999.0],
        }
    )

    out = top_n_by_position(df, n=1)
    assert out.shape[0] == 2  # top 1 QB + top 1 RB
    assert set(out["player"].to_list()) == {"b", "d"}


def test_top_n_by_position_validates_inputs():
    df = pl.DataFrame({"position": ["QB"], "fantasy_points": [1.0]})

    with pytest.raises(ValueError):
        top_n_by_position(df.drop("position"))

    with pytest.raises(ValueError):
        top_n_by_position(df.drop("fantasy_points"))

    with pytest.raises(ValueError):
        top_n_by_position(df, n=0)