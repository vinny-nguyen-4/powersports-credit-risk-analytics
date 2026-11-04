import numpy as np
import pandas as pd
import pytest

from credit_risk import forecast, scorecard, strategy


@pytest.fixture(scope="module")
def model(cfg, portfolio):
    df = scorecard.build_target(portfolio["loans"], portfolio["performance"], cfg)
    sc, iv = scorecard.train_scorecard(df, cfg)
    df["score"] = sc.score(df)
    return df, sc, iv


def test_ks_and_psi_helpers():
    y = np.array([1, 1, 0, 0])
    assert scorecard.ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    x = np.random.default_rng(0).normal(size=5000)
    assert scorecard.psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert scorecard.psi(x, x + 1.0) > 0.25


def test_scorecard_discriminates_out_of_time(model):
    df, sc, _ = model
    metrics = scorecard.evaluate(sc, df).set_index("sample")
    assert metrics.loc["oot", "auc"] > 0.70
    assert metrics.loc["oot", "psi_vs_dev"] < 0.10
    assert "fico" in sc.features


def test_higher_score_means_lower_risk(model):
    df, _, _ = model
    q = pd.qcut(df["score"], 5, labels=False, duplicates="drop")
    bad_by_q = df.groupby(q)["bad"].mean()
    assert bad_by_q.iloc[0] > bad_by_q.iloc[-1] * 3


def test_points_sum_to_score(model):
    df, sc, _ = model
    pts = sc.points_table()
    row = df.iloc[[0]]
    total = 0.0
    for f in sc.features:
        woe = scorecard.apply_bins(row, sc.bins[f]).iloc[0]
        total += pts.loc[(pts["feature"] == f) & np.isclose(pts["woe"], woe), "points"].iloc[0]
    assert abs(total - sc.score(row)[0]) <= len(sc.features)  # rounding per attribute


def test_cutoff_table_monotonic(model, portfolio):
    df, _, _ = model
    perf = portfolio["performance"]
    nl = (perf["chargeoff_amt"] - perf["recovery_amt"]).groupby(perf["loan_id"]).sum()
    econ = strategy.loan_economics(df, nl)
    cuts = strategy.cutoff_table(econ)
    assert cuts["approval_rate"].is_monotonic_decreasing
    assert cuts["approval_rate"].iloc[0] == 1.0
    rec = strategy.recommend_cutoff(cuts)
    assert rec["approval_rate"] >= 0.85


def test_forecasts_are_reasonable(con, cfg, portfolio):
    loans, perf = portfolio["loans"], portfolio["performance"]
    bt = forecast.backtest(con, loans, perf, cfg)
    assert len(bt) == cfg.forecast_horizon
    for col in ("vintage_curve", "roll_rate"):
        err = bt[col].sum() / bt["actual"].sum() - 1
        assert abs(err) < 0.35, f"{col} backtest error {err:.1%}"
