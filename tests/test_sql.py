import pytest

from credit_risk import db


def test_delinquency_trend_matches_pandas(con, cfg, portfolio):
    trend = db.run_sql(con, "delinquency_trend", cfg)
    perf = portfolio["performance"]
    last = perf[perf["month"] == perf["month"].max()]
    row = trend.iloc[-1]
    assert row["ending_balance"] == pytest.approx(last["balance"].sum())
    dq = last.loc[last["status"].isin([1, 2, 3]), "balance"].sum() / last["balance"].sum()
    assert row["dq30_plus_rate"] == pytest.approx(dq)
    assert (trend["dq60_plus_rate"] <= trend["dq30_plus_rate"] + 1e-12).all()


def test_vintage_curves_are_cumulative(con, cfg):
    vint = db.run_sql(con, "vintage_curves", cfg)
    # recoveries are booked with the charge-off, so cumulative net loss never decreases
    diffs = vint.sort_values(["vintage", "mob"]).groupby("vintage")["cum_net_loss_rate"].diff().dropna()
    assert (diffs >= -1e-12).all()


def test_roll_rates_rows_sum_to_at_most_one(con, cfg):
    rr = db.run_sql(con, "roll_rates", cfg, params={"as_of": "2026-09-01", "lookback": 6})
    sums = rr.groupby("from_status")["roll_rate"].sum()
    assert set(sums.index) == {0, 1, 2, 3}
    assert (sums <= 1.0 + 1e-9).all() and (sums > 0.5).all()


def test_segmentation_totals_and_whitelist(con, cfg, portfolio):
    seg = db.segmentation(con, "fico_band", "2026-09", cfg)
    assert seg["loans_originated"].sum() == len(portfolio["loans"])
    with pytest.raises(ValueError):
        db.segmentation(con, "fico; DROP TABLE loans", "2026-09", cfg)
