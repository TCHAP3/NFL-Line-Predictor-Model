"""Load and cache NFL schedule/results/odds data from nflverse (via nfl_data_py)."""
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "schedules.parquet"

FIRST_SEASON = 2002  # modern-era realignment; keeps team codes stable


def _current_season() -> int:
    today = pd.Timestamp.today()
    # NFL season "year" rolls over in March; a season spans Sep(Y)-Feb(Y+1)
    return today.year if today.month >= 3 else today.year - 1


def fetch_schedules(refresh: bool = False) -> pd.DataFrame:
    """Return all regular/postseason games from FIRST_SEASON through the
    current season, including closing lines and final scores where played.
    Cached locally as parquet; pass refresh=True to re-download.
    """
    if CACHE_PATH.exists() and not refresh:
        df = pd.read_parquet(CACHE_PATH)
    else:
        seasons = list(range(FIRST_SEASON, _current_season() + 1))
        df = nfl.import_schedules(seasons)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH)

    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return df


def played_games(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["home_score"].notna() & df["away_score"].notna()].copy()


def upcoming_games(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    out = df[(df["season"] == season) & (df["week"] == week)].copy()
    return out
