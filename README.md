# NFL Game Line Prediction System

Predicts the point spread and total (over/under) for NFL games and compares
the predictions against actual closing market lines, using only public data.

## How it works

1. **Data** (`src/data.py`): pulls every game since 2002 — schedule, final
   score, and closing betting lines (spread, total, moneyline) — from the
   [nflverse](https://github.com/nflverse) data releases via `nfl_data_py`,
   and caches it locally as parquet.

2. **Elo ratings** (`src/elo.py`): a 538-style Elo rating per team, updated
   game-by-game with a margin-of-victory multiplier and home-field
   adjustment, regressed toward the mean between seasons. Only the
   *pre-game* rating is ever used as a feature, so there's no lookahead.

3. **Features** (`src/features.py`): for each game —
   - Elo differential
   - Rolling (last-5-game, shifted) points-for/points-against for each team
   - Rest-day differential, divisional-game flag
   - Dome/outdoor, temperature, wind
   - Starting-QB-changed flag (from the previous game for that team)

4. **Model** (`src/model.py`): two regressors (choose `ridge` or `gbm`) —
   one predicts the home-away scoring margin, one predicts the total points.
   Predictions are compared against the market's `spread_line`/`total_line`
   to produce an "edge" and a pick.

5. **Backtest** (`src/backtest.py`): walk-forward by season — a model
   evaluated on season S is trained only on seasons before S, so it never
   sees that season's results (or any later one) during training.

## Important: what the backtest actually shows

Run `python cli.py backtest` yourself, but the honest headline is:

- **ATS win rate: ~50-52%** (breakeven against standard -110 vig is 52.4%)
- **Margin/total MAE roughly matches, but doesn't beat, the market's own
  closing-line accuracy**

In other words, this system does **not** demonstrate a profitable edge over
closing lines — which is the expected, credible result for a model built on
public pre-game data. NFL closing lines are famously efficient. Treat the
output as an independent, data-driven estimate for comparison purposes, not
as a source of real betting value. This is not financial advice, and past
backtest performance is not a guarantee of anything going forward.

## Setup

```bash
cd nfl-line-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# (re)download the latest schedules/scores/lines
python cli.py update-data

# walk-forward backtest against closing lines
python cli.py backtest --model ridge      # or --model gbm
python cli.py backtest --start-season 2015

# predict a specific week (works for future weeks with no result yet)
python cli.py predict --season 2026 --week 1 --model ridge
```

`predict` output columns:

| column | meaning |
|---|---|
| `pred_spread_home` | model's predicted home margin, in the same sign convention as `spread_line` (positive = home favored) |
| `spread_edge` | `pred_spread_home - spread_line`; positive means the model likes the home side relative to the market |
| `ats_pick` | team the model favors against the spread |
| `pred_total` / `total_line` / `total_edge` / `ou_pick` | same idea, for the total |

## Notes on the data's sign convention

`spread_line` from nflverse is **positive when the home team is favored**
(e.g. `spread_line=27.0` for a Broncos team that blew out the Jaguars at
home) — the opposite of the traditional "-3.5 favorite" gambling notation.
This was verified directly against raw data before wiring up the backtest,
since getting it backwards silently inverts every pick.

## Extending it

- Swap in play-by-play-derived features (EPA/play, success rate, etc. via
  `nfl_data_py.import_pbp_data`) for a stronger signal than box-score
  rolling stats.
- Add injury reports / betting market movement (opening vs. closing line)
  as features.
- Try model stacking or a proper walk-forward hyperparameter search instead
  of the two fixed model configs in `src/model.py`.
