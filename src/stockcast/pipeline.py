"""End-to-end run: data -> features -> backtest -> holdout -> inventory simulation -> report.

Usage: python -m stockcast.pipeline [--out reports]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import plots
from .data import generate_sales
from .evaluation import backtest, interval_coverage, metrics
from .features import KEYS, make_features
from .inventory import forecast_policy, simulate, static_policy
from .models import GradientBoosting, MovingAverage, SeasonalNaive

HOLDOUT_DAYS = 91
LEAD_TIME, REVIEW = 3, 4  # LEAD_TIME + REVIEW <= HORIZON keeps the simulation leakage-free
SERVICE_TARGETS = [0.85, 0.90, 0.95, 0.98, 0.99]


def run(out: Path, seed: int = 7) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw = generate_sales(seed=seed)
    data = make_features(raw)
    last = data["date"].max()
    holdout_start = last - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    train_cutoff = holdout_start - pd.Timedelta(days=1)
    min_train = data["date"].min() + pd.Timedelta(days=28)
    print(f"{len(raw):,} rows, {raw.groupby(KEYS, observed=True).ngroups} series, "
          f"{raw['date'].min().date()} -> {last.date()}")

    # 1. Rolling-origin backtest on the period before the holdout.
    factories = [SeasonalNaive, MovingAverage, lambda: GradientBoosting(seed=seed)]
    fold_metrics, oos = backtest(data, factories, end=train_cutoff, n_folds=4, fold_days=28, min_train_date=min_train)
    summary = fold_metrics.groupby("model")[["wape", "mae", "rmse", "bias"]].mean()
    print("\nBacktest (mean over 4 folds):\n", summary.round(3))

    # Out-of-sample error std per series feeds the safety-stock calculation.
    sigma = {
        name: oos.assign(err=oos["units"] - oos[name]).groupby(KEYS, observed=True)["err"].std()
        for name in ("gbm", "moving_average_28")
    }

    # 2. Final models trained on everything before the holdout.
    train = data[(data["date"] >= min_train) & (data["date"] <= train_cutoff)]
    hold = data[data["date"] >= holdout_start].copy()
    for model in (GradientBoosting(seed=seed), GradientBoosting(quantile=0.1, seed=seed),
                  GradientBoosting(quantile=0.9, seed=seed), SeasonalNaive(), MovingAverage()):
        hold[model.name] = model.fit(train).predict(hold)
    holdout_metrics = {m: metrics(hold["units"], hold[m]) for m in ("seasonal_naive", "moving_average_28", "gbm")}
    coverage = interval_coverage(hold["units"], hold["gbm_q10"], hold["gbm_q90"])
    print("\nHoldout:", {k: round(v["wape"], 3) for k, v in holdout_metrics.items()}, f"80% PI coverage={coverage:.3f}")

    # 3. Inventory simulation on the holdout (last HORIZON days dropped: no forecast covers them).
    sim_days = HOLDOUT_DAYS - (LEAD_TIME + REVIEW)
    rows = []
    for (store, sku), g in hold.groupby(KEYS, observed=True):
        g = g.sort_values("date")
        demand = g["units"].to_numpy()[:sim_days]
        price = float(g["base_price"].iloc[0])
        history = raw[(raw["store"] == store) & (raw["sku"] == sku) & (raw["date"] <= train_cutoff)]["units"]
        history = history.to_numpy()[-182:]
        for target in SERVICE_TARGETS:
            z = NormalDist().inv_cdf(target)
            policies = {
                "Static (mean + z·σ)": static_policy(history, sim_days, LEAD_TIME, REVIEW, z),
                "Forecast: 28-day MA": forecast_policy(g["moving_average_28"].to_numpy(),
                                                       sigma["moving_average_28"][(store, sku)], LEAD_TIME, REVIEW, z),
                "Forecast: GBM": forecast_policy(g["gbm"].to_numpy(), sigma["gbm"][(store, sku)],
                                                 LEAD_TIME, REVIEW, z),
            }
            for policy, (rop, upto) in policies.items():
                rop, upto = rop[:sim_days], upto[:sim_days]
                r = simulate(demand, rop, upto, lead_time=LEAD_TIME, initial_on_hand=upto[0], unit_price=price)
                rows.append({"policy": policy, "target_service": target, "demand": r.demand, "sold": r.sold,
                             "inventory_value": r.avg_on_hand * price, "cost": r.cost, "orders": r.orders})
    inv = (
        pd.DataFrame(rows)
        .groupby(["policy", "target_service"], sort=False)
        .agg(demand=("demand", "sum"), sold=("sold", "sum"), avg_inventory_value=("inventory_value", "sum"),
             total_cost=("cost", "sum"), orders=("orders", "sum"))
        .reset_index()
    )
    inv["fill_rate"] = inv["sold"] / inv["demand"]
    print("\nInventory simulation:\n", inv[["policy", "target_service", "fill_rate", "avg_inventory_value",
                                            "total_cost"]].round(3).to_string(index=False))

    # 4. Artifacts.
    plots.wape_by_model(summary, out / "backtest_wape.png")
    top = hold.groupby(KEYS, observed=True)["units"].sum().idxmax()
    ex = hold[(hold["store"] == top[0]) & (hold["sku"] == top[1])].sort_values("date")
    plots.example_forecast(ex, out / "holdout_forecast.png",
                           f"Holdout forecast, {top[0]} / {top[1]} (highest-volume series)")
    plots.service_tradeoff(inv, out / "service_tradeoff.png")
    fold_metrics.to_csv(out / "backtest_folds.csv", index=False)
    inv.to_csv(out / "inventory_simulation.csv", index=False)

    results = {
        "backtest_mean": summary.round(4).to_dict(orient="index"),
        "holdout": {k: {m: round(v, 4) for m, v in d.items()} for k, d in holdout_metrics.items()},
        "holdout_interval_coverage_80": round(coverage, 4),
        "inventory": inv.round(4).to_dict(orient="records"),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    np.seterr(all="ignore")
    run(args.out, args.seed)


if __name__ == "__main__":
    main()
