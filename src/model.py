"""Train margin/total regressors and turn predictions into line-vs-market picks."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.features import FEATURE_COLUMNS

MODEL_BUILDERS = {
    "ridge": lambda: Ridge(alpha=5.0),
    "gbm": lambda: HistGradientBoostingRegressor(
        max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=20, random_state=0
    ),
}


@dataclass
class GamePredictor:
    kind: str = "ridge"

    def __post_init__(self):
        self.margin_model = MODEL_BUILDERS[self.kind]()
        self.total_model = MODEL_BUILDERS[self.kind]()

    def fit(self, train_df: pd.DataFrame) -> "GamePredictor":
        X = train_df[FEATURE_COLUMNS]
        self.margin_model.fit(X, train_df["margin"])
        self.total_model.fit(X, train_df["total_points"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS]
        out = df.copy()
        out["pred_margin"] = self.margin_model.predict(X)
        out["pred_total"] = self.total_model.predict(X)
        # nflverse's spread_line convention: positive => home team favored by that many
        # points (i.e. it's directly comparable to predicted home margin, no sign flip).
        out["pred_spread_home"] = out["pred_margin"]
        return out


def add_market_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Given predictions + market lines, compute edges and suggested picks."""
    out = df.copy()
    out["spread_edge"] = out["pred_margin"] - out["spread_line"]  # >0 => model favors home side vs market
    out["total_edge"] = out["pred_total"] - out["total_line"]

    out["ats_pick"] = np.where(
        out["spread_edge"].isna(), None, np.where(out["spread_edge"] > 0, out["home_team"], out["away_team"])
    )
    out["ou_pick"] = np.where(
        out["total_edge"].isna(), None, np.where(out["total_edge"] > 0, "OVER", "UNDER")
    )
    return out
