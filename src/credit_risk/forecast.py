"""12-month net charge-off forecasts for the existing book, plus a backtest.

Two independent methods, as a risk team would run them side by side:

1. **Vintage loss-curve method** – average incremental net-loss rate by month on
   book (amount-weighted across vintages), applied to each open vintage's
   remaining life, with a recent-performance overlay that scales the curve to
   the last 6 months of actual losses.
2. **Roll-rate (Markov) method** – dollar-weighted transition matrix between
   delinquency buckets over the last N months, applied to today's balances.

Both forecast run-off of loans already on the books (no new originations).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import CONFIG, Config
from .db import run_sql


def _months_between(a: pd.Series | pd.Timestamp, b: pd.Timestamp):
    return (b.year - a.dt.year) * 12 + (b.month - a.dt.month) if isinstance(a, pd.Series) \
        else (b.year - a.year) * 12 + (b.month - a.month)


def loss_curve(loans: pd.DataFrame, perf: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Amount-weighted average incremental net-loss rate by MOB, observed up to ``as_of``."""
    p = perf[perf["month"] <= as_of]
    inc = (p.assign(net=p["chargeoff_amt"] - p["recovery_amt"])
             .merge(loans[["loan_id", "orig_month"]], on="loan_id")
             .groupby(["orig_month", "mob"])["net"].sum().unstack(fill_value=0.0))
    orig_amt = loans[loans["orig_month"] < as_of].groupby("orig_month")["amount_financed"].sum()
    inc = inc.reindex(orig_amt.index, fill_value=0.0)
    age = _months_between(pd.Series(orig_amt.index), as_of).to_numpy()
    max_mob = int(age.max())
    curve = {}
    for m in range(1, max_mob + 1):
        observed = age >= m
        losses = inc[m].to_numpy()[observed] if m in inc.columns else np.zeros(observed.sum())
        curve[m] = losses.sum() / orig_amt.to_numpy()[observed].sum()
    curve = pd.Series(curve)
    # Extrapolate the tail with an exponential decay fitted to the last 12 observed MOBs
    tail = curve.iloc[-12:]
    slope, intercept = np.polyfit(tail.index, np.log(np.clip(tail.values, 1e-7, None)), 1)
    slope = min(slope, -0.01)
    ext = {m: np.exp(intercept + slope * m) for m in range(max_mob + 1, max_mob + 90)}
    return pd.concat([curve, pd.Series(ext)])


def vintage_curve_forecast(loans: pd.DataFrame, perf: pd.DataFrame, as_of: str,
                           horizon: int = CONFIG.forecast_horizon, overlay_months: int = 6) -> dict:
    as_of = pd.Timestamp(as_of)
    curve = loss_curve(loans, perf, as_of)
    book = loans[loans["orig_month"] <= as_of].groupby("orig_month")["amount_financed"].sum()
    age = pd.Series(_months_between(pd.Series(book.index), as_of).to_numpy(), index=book.index)
    term = loans[loans["orig_month"] <= as_of].groupby("orig_month")["term_months"].max()

    # Recent-performance overlay: actual vs curve-expected losses over the last N months
    p = perf[(perf["month"] > as_of - pd.DateOffset(months=overlay_months)) & (perf["month"] <= as_of)]
    actual_recent = (p["chargeoff_amt"] - p["recovery_amt"]).sum()
    expected_recent = sum(
        amt * curve.loc[max(a - overlay_months + 1, 1):a].sum() for amt, a in zip(book, age) if a >= 1
    )
    overlay = actual_recent / expected_recent if expected_recent else 1.0

    monthly = np.zeros(horizon)
    for amt, a, t in zip(book, age, term):
        for h in range(1, horizon + 1):
            mob = a + h
            if mob <= t:
                monthly[h - 1] += amt * curve.get(mob, 0.0)
    monthly *= overlay
    months = pd.date_range(as_of + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return {"method": "vintage_curve", "overlay": overlay, "curve": curve,
            "monthly": pd.Series(monthly, index=months), "total": float(monthly.sum())}


def roll_rate_matrix(con: duckdb.DuckDBPyConnection, as_of: str, lookback: int, cfg: Config = CONFIG) -> pd.DataFrame:
    iso = pd.Timestamp(as_of).date().isoformat()
    rr = run_sql(con, "roll_rates", cfg, params={"as_of": iso, "lookback": lookback})
    m = rr.pivot(index="from_status", columns="to_status", values="roll_rate").reindex(
        index=range(4), columns=range(6)).fillna(0.0)
    m.index = ["current", "30dpd", "60dpd", "90+dpd"]
    m.columns = ["current", "30dpd", "60dpd", "90+dpd", "charged_off", "paid_off"]
    return m


def roll_rate_forecast(con: duckdb.DuckDBPyConnection, loans: pd.DataFrame, perf: pd.DataFrame, as_of: str,
                       cfg: Config = CONFIG, horizon: int = CONFIG.forecast_horizon) -> dict:
    as_of_ts = pd.Timestamp(as_of)
    matrix = roll_rate_matrix(con, as_of, cfg.roll_rate_lookback, cfg)
    snap = perf[(perf["month"] == as_of_ts) & (perf["status"] <= 3)]
    bal = snap.groupby("status")["balance"].sum().reindex(range(4), fill_value=0.0).to_numpy().copy()
    bal[0] += loans.loc[loans["orig_month"] == as_of_ts, "amount_financed"].sum()  # booked, no payment yet

    hist = perf[(perf["month"] > as_of_ts - pd.DateOffset(months=12)) & (perf["month"] <= as_of_ts)]
    recovery_rate = hist["recovery_amt"].sum() / hist["chargeoff_amt"].sum()

    M = matrix.to_numpy()
    gross = np.zeros(horizon)
    for h in range(horizon):
        gross[h] = bal @ M[:, 4]
        bal = bal @ M[:, :4]
    net = gross * (1 - recovery_rate)
    months = pd.date_range(as_of_ts + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return {"method": "roll_rate", "matrix": matrix, "recovery_rate": recovery_rate,
            "monthly": pd.Series(net, index=months), "total": float(net.sum())}


def actual_runoff_losses(loans: pd.DataFrame, perf: pd.DataFrame, as_of: str, horizon: int) -> pd.Series:
    """Realized net charge-offs over the horizon from loans booked on or before ``as_of``."""
    as_of = pd.Timestamp(as_of)
    ids = loans.loc[loans["orig_month"] <= as_of, "loan_id"]
    end = as_of + pd.DateOffset(months=horizon)
    p = perf[perf["loan_id"].isin(ids) & (perf["month"] > as_of) & (perf["month"] <= end)]
    return (p["chargeoff_amt"] - p["recovery_amt"]).groupby(p["month"]).sum()


def backtest(con, loans, perf, cfg: Config = CONFIG) -> pd.DataFrame:
    """Forecast as of ``cfg.backtest_as_of`` and compare with what actually happened."""
    vc = vintage_curve_forecast(loans, perf, cfg.backtest_as_of, cfg.forecast_horizon)
    rr = roll_rate_forecast(con, loans, perf, cfg.backtest_as_of, cfg, cfg.forecast_horizon)
    actual = actual_runoff_losses(loans, perf, cfg.backtest_as_of, cfg.forecast_horizon)
    df = pd.DataFrame({"actual": actual, "vintage_curve": vc["monthly"], "roll_rate": rr["monthly"]})
    df.index.name = "month"
    return df.reset_index()
