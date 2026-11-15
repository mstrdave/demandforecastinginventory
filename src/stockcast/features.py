"""Leakage-safe feature engineering for a global 7-day-ahead forecaster.

Every demand-derived feature is lagged by at least ``HORIZON`` days, so a
forecast for day *d* uses only sales observed up to *d - HORIZON*. Price,
promotions and holidays are planned in advance and are used for day *d* itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 7
KEYS = ["store", "sku"]
LAGS = [7, 14, 21, 28, 364]
ROLL_WINDOWS = [7, 28, 91]

FEATURES = [
    "store", "sku", "category",
    *[f"lag_{lag}" for lag in LAGS],
    *[f"rmean_{w}" for w in ROLL_WINDOWS],
    "rstd_28", "ewm_14",
    "dow", "month", "doy_sin", "doy_cos", "dom",
    "promo", "price_ratio", "is_holiday", "days_to_xmas",
]
CATEGORICAL = ["store", "sku", "category"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if not set(LAGS).isdisjoint(range(HORIZON)):
        raise ValueError("lags must be >= HORIZON")
    out = df.sort_values([*KEYS, "date"]).reset_index(drop=True)
    grouped = out.groupby(KEYS, observed=True, sort=False)["units"]
    for lag in LAGS:
        out[f"lag_{lag}"] = grouped.shift(lag)

    base = grouped.shift(HORIZON)
    base_grouped = base.groupby([out["store"], out["sku"]], observed=True, sort=False)
    for w in ROLL_WINDOWS:
        out[f"rmean_{w}"] = base_grouped.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    out["rstd_28"] = base_grouped.transform(lambda s: s.rolling(28, min_periods=2).std())
    out["ewm_14"] = base_grouped.transform(lambda s: s.ewm(span=14, ignore_na=True).mean())

    date = out["date"]
    out["dow"] = date.dt.dayofweek
    out["month"] = date.dt.month
    out["dom"] = date.dt.day
    doy = date.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["price_ratio"] = out["price"] / out["base_price"]
    xmas = pd.to_datetime(date.dt.year.astype(str) + "-12-25")
    out["days_to_xmas"] = (xmas - date).dt.days.where(lambda d: d >= 0, 365)
    return out
