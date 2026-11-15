"""Report figures (static PNGs for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(width=8, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def wape_by_model(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = _fig(7, 3.6)
    s = summary.sort_values("wape", ascending=False)
    y = np.arange(len(s))
    ax.barh(y, s["wape"] * 100, color=[SERIES[0] if m == "gbm" else "#a9a8a2" for m in s.index], height=0.55)
    ax.set_yticks(y, s.index)
    for yi, v in zip(y, s["wape"] * 100, strict=True):
        ax.text(v + 0.4, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    _style(ax, "Backtest WAPE, 7-day-ahead (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("WAPE (%)", color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def example_forecast(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = _fig()
    ax.fill_between(frame["date"], frame["gbm_q10"], frame["gbm_q90"], color=SERIES[0], alpha=0.15,
                    linewidth=0, label="80% interval")
    ax.plot(frame["date"], frame["units"], color=INK2, linewidth=1.2, marker="o", markersize=2.5, label="Actual")
    ax.plot(frame["date"], frame["gbm"], color=SERIES[0], linewidth=2, label="Forecast")
    _style(ax, title)
    ax.set_ylabel("Units / day", color=INK2)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def service_tradeoff(inv: pd.DataFrame, path: Path) -> None:
    fig, ax = _fig(7, 4.2)
    for color, (policy, g) in zip(SERIES, inv.groupby("policy", sort=False), strict=False):
        g = g.sort_values("target_service")
        ax.plot(g["avg_inventory_value"] / 1000, g["fill_rate"] * 100, color=color, linewidth=2,
                marker="o", markersize=6, label=policy)
        last = g.iloc[-1]
        ax.annotate(policy, (last["avg_inventory_value"] / 1000, last["fill_rate"] * 100),
                    xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    _style(ax, "Fill rate vs. inventory held (up-and-left is better)")
    ax.set_xlabel("Average inventory value (thousands)", color=INK2)
    ax.set_ylabel("Fill rate (%)", color=INK2)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
