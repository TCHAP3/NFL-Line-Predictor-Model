"""538-style Elo ratings computed causally (pre-game rating only uses past games)."""
import math

import pandas as pd

INITIAL_ELO = 1500.0
HOME_ADV = 55.0       # elo points added to home team's rating for win-prob purposes
K = 20.0
SEASON_REGRESSION = 1.0 / 3.0  # fraction reverted toward 1500 between seasons


def _expected(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """games must be sorted chronologically. Returns a copy with home_elo_pre,
    away_elo_pre, home_elo_post, away_elo_post columns."""
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    rating: dict[str, float] = {}
    last_season: dict[str, int] = {}

    home_pre, away_pre, home_post, away_post = [], [], [], []

    for row in games.itertuples(index=False):
        season = row.season
        home, away = row.home_team, row.away_team

        for team in (home, away):
            if team not in rating:
                rating[team] = INITIAL_ELO
                last_season[team] = season
            elif last_season[team] != season:
                rating[team] = INITIAL_ELO + (rating[team] - INITIAL_ELO) * (1 - SEASON_REGRESSION)
                last_season[team] = season

        h_elo, a_elo = rating[home], rating[away]
        home_pre.append(h_elo)
        away_pre.append(a_elo)

        home_score, away_score = row.home_score, row.away_score
        if pd.isna(home_score) or pd.isna(away_score):
            home_post.append(h_elo)
            away_post.append(a_elo)
            continue

        elo_diff = (h_elo + HOME_ADV) - a_elo
        expected_home = _expected(elo_diff)
        margin = home_score - away_score

        if margin > 0:
            actual_home = 1.0
        elif margin < 0:
            actual_home = 0.0
        else:
            actual_home = 0.5

        winner_elo_diff = elo_diff if margin >= 0 else -elo_diff
        mov_mult = math.log(abs(margin) + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))
        mov_mult = max(mov_mult, 0.5)  # guard against pathological blowout+underdog combos

        shift = K * mov_mult * (actual_home - expected_home)
        rating[home] = h_elo + shift
        rating[away] = a_elo - shift

        home_post.append(rating[home])
        away_post.append(rating[away])

    out = games.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["home_elo_post"] = home_post
    out["away_elo_post"] = away_post
    return out


def current_ratings(games_with_elo: pd.DataFrame) -> dict:
    """Latest post-game elo for every team, from a frame already run through compute_elo."""
    ratings = {}
    for row in games_with_elo.itertuples(index=False):
        ratings[row.home_team] = row.home_elo_post
        ratings[row.away_team] = row.away_elo_post
    return ratings
