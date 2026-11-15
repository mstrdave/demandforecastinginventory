"""Discrete-time inventory simulation for reorder-point / order-up-to policies.

Each day, in order: receive orders that arrive today, serve demand from stock
(demand that can't be served is lost), then review the inventory position
(on hand + on order) and place an order if it is at or below the reorder point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Costs:
    holding_per_unit_day: float = 0.02  # fraction of unit price per day
    lost_sale_per_unit: float = 0.5  # fraction of unit price (lost margin + goodwill)
    order_fixed: float = 5.0  # currency per order placed


@dataclass
class SimResult:
    demand: float
    sold: float
    lost: float
    avg_on_hand: float
    stockout_days: int
    orders: int
    cost: float

    @property
    def fill_rate(self) -> float:
        return self.sold / self.demand if self.demand else 1.0


def simulate(
    demand: np.ndarray,
    reorder_point: np.ndarray,
    order_up_to: np.ndarray,
    *,
    lead_time: int,
    initial_on_hand: float,
    unit_price: float = 1.0,
    costs: Costs = Costs(),
) -> SimResult:
    demand = np.asarray(demand, float)
    n = len(demand)
    if not (len(reorder_point) == len(order_up_to) == n):
        raise ValueError("policy arrays must match demand length")
    if lead_time < 1:
        raise ValueError("lead_time must be >= 1")

    on_hand = float(initial_on_hand)
    pipeline: deque[tuple[int, float]] = deque()
    sold = lost = on_hand_sum = 0.0
    stockout_days = orders = 0

    for t in range(n):
        while pipeline and pipeline[0][0] == t:
            on_hand += pipeline.popleft()[1]
        served = min(on_hand, demand[t])
        on_hand -= served
        sold += served
        if served < demand[t]:
            lost += demand[t] - served
            stockout_days += 1
        on_hand_sum += on_hand

        position = on_hand + sum(q for _, q in pipeline)
        if position <= reorder_point[t]:
            qty = float(np.ceil(order_up_to[t] - position))
            if qty > 0:
                pipeline.append((t + lead_time, qty))
                orders += 1

    avg_on_hand = on_hand_sum / n if n else 0.0
    cost = (
        costs.holding_per_unit_day * unit_price * on_hand_sum
        + costs.lost_sale_per_unit * unit_price * lost
        + costs.order_fixed * orders
    )
    return SimResult(float(demand.sum()), sold, lost, avg_on_hand, stockout_days, orders, cost)


def static_policy(history: np.ndarray, n_days: int, lead_time: int, review: int, z: float):
    """Classic textbook policy: ROP = mu*L + z*sigma*sqrt(L), from recent history."""
    mu, sd = float(np.mean(history)), float(np.std(history, ddof=1))
    rop = mu * lead_time + z * sd * np.sqrt(lead_time)
    return np.full(n_days, rop), np.full(n_days, rop + mu * review)


def forecast_policy(forecast: np.ndarray, sigma: float, lead_time: int, review: int, z: float):
    """Forecast-driven policy: the reorder point follows expected lead-time demand.

    ``forecast[t]`` is the forecast for day t. At day t, the next ``lead_time``
    days' forecasts cover the lead time and the ``review`` days after that set
    the order-up-to level. ``sigma`` is the out-of-sample forecast error std.
    """
    f = np.asarray(forecast, float)
    n = len(f)
    padded = np.concatenate([f, np.full(lead_time + review, f[-lead_time - review:].mean())])
    csum = np.concatenate([[0.0], np.cumsum(padded)])
    lt_demand = csum[np.arange(n) + 1 + lead_time] - csum[np.arange(n) + 1]
    rv_demand = csum[np.arange(n) + 1 + lead_time + review] - csum[np.arange(n) + 1 + lead_time]
    rop = lt_demand + z * sigma * np.sqrt(lead_time)
    return rop, rop + rv_demand
