"""Generate predictions for a specific season/week, including games with no
market line yet posted (spread/total will just show as NaN in that case)."""
import pandas as pd

from src.model import GamePredictor, add_market_comparison

DISPLAY_COLUMNS = [
    "game_id", "gameday", "away_team", "home_team",
    "pred_spread_home", "spread_line", "spread_edge", "ats_pick",
    "pred_total", "total_line", "total_edge", "ou_pick",
]


def predict_week(dataset: pd.DataFrame, season: int, week: int, kind: str = "ridge") -> pd.DataFrame:
    train_df = dataset[
        (dataset["season"] < season) | ((dataset["season"] == season) & (dataset["week"] < week))
    ].dropna(subset=["margin", "total_points"])

    target_df = dataset[(dataset["season"] == season) & (dataset["week"] == week)].copy()
    if target_df.empty:
        raise ValueError(f"No games found for season={season} week={week}")

    model = GamePredictor(kind=kind).fit(train_df)
    preds = model.predict(target_df)
    preds = add_market_comparison(preds)

    for col in ("spread_line", "total_line"):
        preds[col] = preds[col]  # may be NaN if book hasn't posted yet; leave as-is

    return preds[DISPLAY_COLUMNS].sort_values("gameday").reset_index(drop=True)
