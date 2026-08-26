#!/usr/bin/env python3
"""NFL game line prediction system.

Examples:
    python cli.py update-data
    python cli.py backtest --model gbm
    python cli.py predict --season 2025 --week 4
"""
import argparse
import sys

import pandas as pd

from src.backtest import run_backtest
from src.data import fetch_schedules, played_games
from src.features import build_dataset
from src.predict import predict_week

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda v: f"{v:.1f}")


def cmd_update_data(args):
    df = fetch_schedules(refresh=True)
    print(f"Fetched {len(df)} games, seasons {df['season'].min()}-{df['season'].max()}")


def cmd_backtest(args):
    raw = fetch_schedules()
    dataset = build_dataset(raw)
    result = run_backtest(dataset, kind=args.model, start_season=args.start_season)

    overall = result["overall"]
    print("\n=== Backtest summary ({} model) ===".format(args.model))
    print(f"Games graded:         {overall['n_games']}")
    print(f"ATS win %:            {overall['ats_win_pct']:.3f}  (breakeven at -110 = {overall['breakeven_ats']})")
    print(f"O/U win %:            {overall['ou_win_pct']:.3f}")
    print(f"Model margin MAE:     {overall['margin_mae']:.2f}  (market MAE: {overall['market_margin_mae']:.2f})")
    print(f"Model total MAE:      {overall['total_mae']:.2f}  (market MAE: {overall['market_total_mae']:.2f})")

    print("\n=== By season ===")
    print(result["by_season"].to_string(index=False))


def cmd_predict(args):
    raw = fetch_schedules()
    dataset = build_dataset(raw)
    preds = predict_week(dataset, season=args.season, week=args.week, kind=args.model)
    print(f"\n=== Predictions: {args.season} Week {args.week} ({args.model} model) ===")
    print(preds.to_string(index=False))


def cmd_next(args):
    raw = fetch_schedules(refresh=True)
    unplayed = raw[raw["home_score"].isna()]
    if unplayed.empty:
        print("No upcoming games found in the cached schedule (season may be fully complete).")
        return
    season, week = unplayed.sort_values(["season", "week"])[["season", "week"]].iloc[0]
    season, week = int(season), int(week)

    dataset = build_dataset(raw)
    preds = predict_week(dataset, season=season, week=week, kind=args.model)
    print(f"\n=== Predictions: {season} Week {week} ({args.model} model) ===")
    print(preds.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser("update-data", help="Re-download schedules/odds from nflverse")
    p_update.set_defaults(func=cmd_update_data)

    p_bt = sub.add_parser("backtest", help="Walk-forward backtest against closing lines")
    p_bt.add_argument("--model", choices=["ridge", "gbm"], default="ridge")
    p_bt.add_argument("--start-season", type=int, default=None)
    p_bt.set_defaults(func=cmd_backtest)

    p_pred = sub.add_parser("predict", help="Predict spread/total for a given season+week")
    p_pred.add_argument("--season", type=int, required=True)
    p_pred.add_argument("--week", type=int, required=True)
    p_pred.add_argument("--model", choices=["ridge", "gbm"], default="ridge")
    p_pred.set_defaults(func=cmd_predict)

    p_next = sub.add_parser("next", help="Refresh data and predict whatever the next unplayed week is")
    p_next.add_argument("--model", choices=["ridge", "gbm"], default="ridge")
    p_next.set_defaults(func=cmd_next)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
