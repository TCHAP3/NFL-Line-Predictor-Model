"""Walk-forward backtest: train on all seasons before S, evaluate on season S.

This avoids lookahead — a model evaluated on 2018 never saw a single 2018
result (or later) during training, and within a season the rolling/elo
features for any given game only reflect games played earlier that season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import GamePredictor, add_market_comparison

MIN_TRAIN_SEASONS = 3
BREAKEVEN_ATS = 0.5238  # win rate needed to beat -110 vig


def _grade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # spread_line: positive => home favored by that many points (nflverse convention)
    df["cover_margin"] = df["margin"] - df["spread_line"]
    df["ats_result"] = np.select(
        [df["cover_margin"] > 0, df["cover_margin"] < 0], ["home", "away"], default="push"
    )
    winning_side_team = np.where(df["ats_result"] == "home", df["home_team"], df["away_team"])
    df["ats_correct"] = (df["ats_pick"].to_numpy() == winning_side_team).astype(float)
    df.loc[df["ats_result"] == "push", "ats_correct"] = np.nan

    df["total_result"] = np.select(
        [df["total_points"] > df["total_line"], df["total_points"] < df["total_line"]],
        ["over", "under"], default="push",
    )
    df["ou_correct"] = np.where(
        df["total_result"] == "push", np.nan,
        (df["ou_pick"].str.lower() == df["total_result"]).astype(float),
    )
    return df


def run_backtest(dataset: pd.DataFrame, kind: str = "ridge", start_season: int | None = None) -> dict:
    dataset = dataset.dropna(subset=["margin", "total_points", "spread_line", "total_line"]).copy()
    seasons = sorted(dataset["season"].unique())
    if start_season is None:
        start_season = seasons[MIN_TRAIN_SEASONS]

    rows = []
    per_season = []
    for test_season in [s for s in seasons if s >= start_season]:
        train_df = dataset[dataset["season"] < test_season]
        test_df = dataset[dataset["season"] == test_season]
        if len(train_df) < 100 or test_df.empty:
            continue

        model = GamePredictor(kind=kind).fit(train_df)
        preds = model.predict(test_df)
        preds = add_market_comparison(preds)
        graded = _grade(preds)
        rows.append(graded)

        per_season.append({
            "season": test_season,
            "n_games": len(graded),
            "ats_win_pct": graded["ats_correct"].mean(),
            "ou_win_pct": graded["ou_correct"].mean(),
            "margin_mae": (graded["pred_margin"] - graded["margin"]).abs().mean(),
            "total_mae": (graded["pred_total"] - graded["total_points"]).abs().mean(),
            "market_margin_mae": (graded["spread_line"] - graded["margin"]).abs().mean(),
            "market_total_mae": (graded["total_line"] - graded["total_points"]).abs().mean(),
        })

    all_graded = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    season_summary = pd.DataFrame(per_season)

    overall = {
        "kind": kind,
        "n_games": len(all_graded),
        "ats_win_pct": all_graded["ats_correct"].mean() if len(all_graded) else float("nan"),
        "ou_win_pct": all_graded["ou_correct"].mean() if len(all_graded) else float("nan"),
        "margin_mae": (all_graded["pred_margin"] - all_graded["margin"]).abs().mean() if len(all_graded) else float("nan"),
        "total_mae": (all_graded["pred_total"] - all_graded["total_points"]).abs().mean() if len(all_graded) else float("nan"),
        "market_margin_mae": (all_graded["spread_line"] - all_graded["margin"]).abs().mean() if len(all_graded) else float("nan"),
        "market_total_mae": (all_graded["total_line"] - all_graded["total_points"]).abs().mean() if len(all_graded) else float("nan"),
        "breakeven_ats": BREAKEVEN_ATS,
    }

    return {"overall": overall, "by_season": season_summary, "graded": all_graded}
