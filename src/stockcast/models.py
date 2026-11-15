"""Forecasting models: two baselines and gradient-boosted trees."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, FEATURES


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> Forecaster: ...
    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


class SeasonalNaive:
    """Same weekday last week: the standard benchmark for daily retail demand."""

    name = "seasonal_naive"

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["lag_7"].fillna(frame["rmean_28"]).fillna(0).to_numpy()


class MovingAverage:
    name = "moving_average_28"

    def fit(self, train: pd.DataFrame) -> MovingAverage:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["rmean_28"].fillna(0).to_numpy()


class GradientBoosting:
    """One global model across all series.

    Poisson loss fits non-negative, overdispersed counts better than squared
    error. Quantile mode trains the same model for prediction intervals.
    """

    def __init__(self, loss: str = "poisson", quantile: float | None = None, seed: int = 0) -> None:
        self.loss, self.quantile = loss, quantile
        self.name = "gbm" if quantile is None else f"gbm_q{int(quantile * 100)}"
        kwargs = {"quantile": quantile} if quantile is not None else {}
        self.model = HistGradientBoostingRegressor(
            loss="quantile" if quantile is not None else loss,
            learning_rate=0.06,
            max_iter=400,
            max_leaf_nodes=48,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=CATEGORICAL,
            early_stopping=False,
            random_state=seed,
            **kwargs,
        )

    def fit(self, train: pd.DataFrame) -> GradientBoosting:
        self.model.fit(train[FEATURES], train["units"])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(frame[FEATURES]), 0, None)
