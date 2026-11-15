"""Forecast metrics and rolling-origin backtesting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .models import Forecaster


def metrics(y: np.ndarray, f: np.ndarray) -> dict[str, float]:
    y, f = np.asarray(y, float), np.asarray(f, float)
    err = f - y
    total = max(y.sum(), 1e-9)
    return {
        "wape": float(np.abs(err).sum() / total),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.sum() / total),
    }


def rolling_origin_folds(dates: pd.Series, end: pd.Timestamp, n_folds: int, fold_days: int):
    """Yield (train_end, test_start, test_end) for consecutive test windows ending at ``end``."""
    for i in range(n_folds, 0, -1):
        test_end = end - pd.Timedelta(days=(i - 1) * fold_days)
        test_start = test_end - pd.Timedelta(days=fold_days - 1)
        yield test_start - pd.Timedelta(days=1), test_start, test_end


def backtest(
    data: pd.DataFrame,
    model_factories: list[Callable[[], Forecaster]],
    *,
    end: pd.Timestamp,
    n_folds: int = 4,
    fold_days: int = 28,
    min_train_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit every model once per fold, then score 7-day-ahead forecasts on the fold.

    Returns (per-fold metrics, out-of-sample predictions).
    """
    rows, preds = [], []
    for fold, (train_end, test_start, test_end) in enumerate(
        rolling_origin_folds(data["date"], end, n_folds, fold_days), start=1
    ):
        train = data[(data["date"] <= train_end)]
        if min_train_date is not None:
            train = train[train["date"] >= min_train_date]
        test = data[(data["date"] >= test_start) & (data["date"] <= test_end)]
        fold_pred = test[["date", "store", "sku", "units"]].copy()
        fold_pred["fold"] = fold
        for factory in model_factories:
            model = factory().fit(train)
            f = model.predict(test)
            fold_pred[model.name] = f
            rows.append({"fold": fold, "model": model.name, "test_start": test_start.date(),
                         **metrics(test["units"].to_numpy(), f)})
        preds.append(fold_pred)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y)
    return float(((y >= lower) & (y <= upper)).mean())
