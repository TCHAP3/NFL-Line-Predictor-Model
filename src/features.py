"""Feature engineering: rolling team form, rest, QB continuity, environment.

Everything here is causal — for a given game, only information available
before kickoff is used (rolling stats are shifted so the current game's own
score never leaks into its own features).
"""
import numpy as np
import pandas as pd

from src.elo import compute_elo

ROLL_WINDOW = 5
LEAGUE_AVG_PTS = 22.0

FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "div_game",
    "is_dome",
    "temp_adj",
    "wind_adj",
    "home_off_form",
    "home_def_form",
    "away_off_form",
    "away_def_form",
    "qb_change_home",
    "qb_change_away",
    "week_num",
]


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per team per game, with points for/against and QB id."""
    home = games[["game_id", "gameday", "season", "home_team", "home_score", "away_score", "home_qb_id"]].rename(
        columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against", "home_qb_id": "qb_id"}
    )
    away = games[["game_id", "gameday", "season", "away_team", "away_score", "home_score", "away_qb_id"]].rename(
        columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against", "away_qb_id": "qb_id"}
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    return long


def _rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    long = long.copy()
    grp = long.groupby("team", group_keys=False)

    def _shifted_roll(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()

    long["off_form"] = grp["points_for"].apply(_shifted_roll)
    long["def_form"] = grp["points_against"].apply(_shifted_roll)
    long["off_form"] = long["off_form"].fillna(LEAGUE_AVG_PTS)
    long["def_form"] = long["def_form"].fillna(LEAGUE_AVG_PTS)

    long["prev_qb_id"] = grp["qb_id"].shift(1)
    long["qb_change"] = ((long["prev_qb_id"].notna()) & (long["prev_qb_id"] != long["qb_id"])).astype(int)
    return long


def build_dataset(raw_games: pd.DataFrame) -> pd.DataFrame:
    """Returns raw_games augmented with elo + form + env features, one row per game."""
    games = compute_elo(raw_games)

    long = _team_game_log(games)
    long = _rolling_form(long)

    # each game_id appears twice in `long` (once per team) -- split by matching team to home/away
    home_side = games[["game_id", "home_team"]].merge(
        long[["game_id", "team", "off_form", "def_form", "qb_change"]],
        left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left",
    ).rename(columns={"off_form": "home_off_form", "def_form": "home_def_form", "qb_change": "qb_change_home"})

    away_side = games[["game_id", "away_team"]].merge(
        long[["game_id", "team", "off_form", "def_form", "qb_change"]],
        left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left",
    ).rename(columns={"off_form": "away_off_form", "def_form": "away_def_form", "qb_change": "qb_change_away"})

    out = games.copy()
    out = out.merge(home_side[["game_id", "home_off_form", "home_def_form", "qb_change_home"]], on="game_id", how="left")
    out = out.merge(away_side[["game_id", "away_off_form", "away_def_form", "qb_change_away"]], on="game_id", how="left")

    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    out["div_game"] = out["div_game"].fillna(0).astype(int)
    out["is_dome"] = out["roof"].isin(["dome", "closed"]).astype(int)
    out["temp_adj"] = np.where(out["is_dome"] == 1, 70.0, out["temp"]).astype(float)
    out["temp_adj"] = out["temp_adj"].fillna(60.0)
    out["wind_adj"] = np.where(out["is_dome"] == 1, 0.0, out["wind"]).astype(float)
    out["wind_adj"] = out["wind_adj"].fillna(5.0)
    out["week_num"] = out["week"].clip(upper=18)

    out["margin"] = out["home_score"] - out["away_score"]
    out["total_points"] = out["home_score"] + out["away_score"]

    return out
