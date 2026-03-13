import nflreadpy as nfl
import polars as pl

df = nfl.load_player_stats(2025, summary_level="week")
# print(df.columns)
# print(df.head(3))
print(df.filter(pl.col("position").is_in(["RB","WR"])).select([
  "player_display_name","position","week","receiving_yards","receiving_tds","rushing_tds",
  "rushing_fumbles_lost","receiving_fumbles_lost","sack_fumbles_lost",
  "fg_made","pat_made"
]).sort("receiving_yards", descending=True).head(10))
