"""Synthetic retail demand with the structure real sales data has.

Each (store, sku) series combines a trend, weekly and yearly seasonality,
promotions with price cuts and their price elasticity, US federal holidays, and
negative-binomial noise (overdispersed counts, including slow, intermittent
SKUs). The data is synthetic so the repo runs offline, and because the
data-generating process is known, model behaviour can be checked against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CATEGORIES = {
    # category: (yearly amplitude, yearly phase, trend per year, promo lift)
    "grocery": (0.10, 0.0, 0.03, 0.35),
    "household": (0.15, 1.5, 0.05, 0.45),
    "outdoor": (0.55, -1.6, 0.08, 0.60),  # peaks in summer
    "electronics": (0.35, 1.9, 0.10, 0.80),  # peaks around the holidays
}
WEEKLY = np.array([-0.10, -0.15, -0.12, -0.05, 0.10, 0.25, 0.07])  # Mon..Sun


def generate_sales(
    n_stores: int = 5,
    n_skus: int = 20,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    t_years = np.arange(n_days) / 365.25
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    holidays = USFederalHolidayCalendar().holidays(dates.min(), dates.max())
    is_holiday = dates.isin(holidays).astype(int)
    # Shopping lift in the days before Christmas.
    pre_xmas = ((dates.month == 12) & (dates.day >= 15) & (dates.day <= 24)).astype(int)

    store_mult = rng.uniform(0.6, 1.6, n_stores)
    cats = list(CATEGORIES)
    sku_cat = [cats[i % len(cats)] for i in range(n_skus)]
    sku_base = np.exp(rng.normal(1.6, 0.9, n_skus))  # ~0.8 to ~30 units/day
    sku_price = rng.uniform(2, 80, n_skus).round(2)
    sku_elasticity = rng.uniform(-2.5, -0.8, n_skus)
    dispersion = rng.uniform(2, 10, n_skus)  # NB shape; lower means noisier

    frames = []
    for s in range(n_stores):
        for k in range(n_skus):
            amp, phase, trend, promo_lift = CATEGORIES[sku_cat[k]]
            promo = np.zeros(n_days, dtype=int)
            for start_idx in rng.choice(n_days - 7, size=int(n_days * 0.012), replace=False):
                promo[start_idx : start_idx + rng.integers(3, 8)] = 1
            price = sku_price[k] * np.where(promo == 1, 0.8, 1.0) * rng.normal(1, 0.01, n_days)
            log_mu = (
                np.log(sku_base[k] * store_mult[s])
                + trend * t_years
                + WEEKLY[dow]
                + amp * np.sin(2 * np.pi * doy / 365.25 + phase)
                + promo_lift * promo
                + sku_elasticity[k] * np.log(price / sku_price[k]) * 0.3
                + 0.4 * is_holiday
                + (0.5 if sku_cat[k] == "electronics" else 0.15) * pre_xmas
            )
            mu = np.exp(log_mu)
            lam = rng.gamma(dispersion[k], mu / dispersion[k])
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store": f"store_{s:02d}",
                        "sku": f"sku_{k:03d}",
                        "category": sku_cat[k],
                        "price": price.round(2),
                        "base_price": sku_price[k],
                        "promo": promo,
                        "is_holiday": is_holiday,
                        "units": rng.poisson(lam),
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    for col in ("store", "sku", "category"):
        df[col] = df[col].astype("category")
    return df
