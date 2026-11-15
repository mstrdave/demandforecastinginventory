import numpy as np
import pandas as pd
import pytest

from stockcast.data import generate_sales
from stockcast.evaluation import backtest, interval_coverage, metrics, rolling_origin_folds
from stockcast.features import FEATURES, HORIZON, make_features
from stockcast.inventory import forecast_policy, simulate, static_policy
from stockcast.models import GradientBoosting, SeasonalNaive


@pytest.fixture(scope="module")
def small():
    return generate_sales(n_stores=2, n_skus=4, start="2023-01-01", end="2024-12-31", seed=1)


def test_generated_data_is_well_formed(small):
    assert len(small) == 2 * 4 * 731
    assert (small["units"] >= 0).all()
    assert not small.duplicated(["date", "store", "sku"]).any()
    # Promotions should lift demand: the generator's known effect must be recoverable.
    assert small[small.promo == 1]["units"].mean() > small[small.promo == 0]["units"].mean()


def test_features_have_no_lookahead(small):
    """Changing the most recent HORIZON days of sales must not change any feature for the last day."""
    feats = make_features(small)
    tampered = small.copy()
    last = tampered["date"].max()
    recent = tampered["date"] > last - pd.Timedelta(days=HORIZON)
    tampered.loc[recent, "units"] = 10_000
    feats_t = make_features(tampered)
    numeric = [c for c in FEATURES if c not in ("store", "sku", "category")]
    a = feats[feats["date"] == last][numeric].reset_index(drop=True)
    b = feats_t[feats_t["date"] == last][numeric].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_lag_feature_matches_shift(small):
    feats = make_features(small)
    one = feats[(feats.store == "store_00") & (feats.sku == "sku_000")].sort_values("date")
    np.testing.assert_array_equal(one["lag_7"].to_numpy()[7:], one["units"].to_numpy()[:-7])


def test_metrics_known_values():
    m = metrics(np.array([10, 10]), np.array([12, 6]))
    assert m["wape"] == pytest.approx(6 / 20)
    assert m["bias"] == pytest.approx(-2 / 20)
    assert m["mae"] == pytest.approx(3)
    assert interval_coverage(np.array([1, 5, 9]), np.array([0, 0, 0]), np.array([4, 6, 8])) == pytest.approx(2 / 3)


def test_folds_are_contiguous_and_ordered():
    folds = list(rolling_origin_folds(None, pd.Timestamp("2024-12-31"), 3, 28))
    assert folds[-1][2] == pd.Timestamp("2024-12-31")
    for (train_end, start, end), nxt in zip(folds, folds[1:], strict=False):
        assert train_end < start <= end
        assert nxt[1] == end + pd.Timedelta(days=1)


def test_gbm_beats_seasonal_naive(small):
    feats = make_features(small)
    feats = feats[feats["date"] >= feats["date"].min() + pd.Timedelta(days=28)]
    scores, _ = backtest(feats, [SeasonalNaive, GradientBoosting], end=feats["date"].max(), n_folds=2)
    mean = scores.groupby("model")["wape"].mean()
    assert mean["gbm"] < mean["seasonal_naive"]


def test_simulation_conserves_units():
    rng = np.random.default_rng(0)
    demand = rng.poisson(5, 60)
    rop, upto = static_policy(demand, 60, lead_time=3, review=4, z=1.64)
    r = simulate(demand, rop, upto, lead_time=3, initial_on_hand=upto[0])
    assert r.sold + r.lost == pytest.approx(demand.sum())
    assert 0 <= r.fill_rate <= 1


def test_simulation_hand_checked_case():
    # ROP 0, order-up-to 5, lead time 1: day 0 sells 3 (left 2), orders nothing; day 1 sells 2,
    # loses 2, orders 5 -> arrives day 2, sells 1.
    r = simulate(np.array([3, 4, 1]), np.zeros(3), np.full(3, 5), lead_time=1, initial_on_hand=5)
    assert (r.sold, r.lost, r.orders, r.stockout_days) == (6, 2, 1, 1)


def test_more_safety_stock_never_lowers_fill_rate():
    rng = np.random.default_rng(1)
    demand = rng.poisson(8, 90)
    forecast = np.full(90, 8.0)
    rates = []
    for z in (0.0, 1.0, 2.0, 3.0):
        rop, upto = forecast_policy(forecast, sigma=3.0, lead_time=3, review=4, z=z)
        rates.append(simulate(demand, rop, upto, lead_time=3, initial_on_hand=upto[0]).fill_rate)
    assert rates == sorted(rates)
