"""Decision-engine strategy: score-band economics and cutoff trade-offs.

All loans in the data were booked, so this is a *swap-out* analysis: it shows
what the approved population would have looked like at stricter cutoffs.
Reject inference is out of scope (no declined applicants are simulated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, Config


def loan_economics(df: pd.DataFrame, net_loss: pd.Series, cfg: Config = CONFIG) -> pd.DataFrame:
    """Rough lifetime margin per loan vs realized net loss.

    margin ~= amount * (APR - cost of funds - servicing) * average life * 0.5
    (the 0.5 approximates the average outstanding balance of an amortizing loan).
    """
    out = df.copy()
    spread = np.clip(out["apr"] - cfg.cost_of_funds_pct - cfg.servicing_cost_pct, 0, None) / 100
    out["est_margin"] = out["amount_financed"] * spread * cfg.avg_life_years * 0.5
    out["net_loss"] = out["loan_id"].map(net_loss).fillna(0.0)
    out["est_profit"] = out["est_margin"] - out["net_loss"]
    return out


def score_bands(df: pd.DataFrame, width: int = 20) -> pd.DataFrame:
    lo = int(np.floor(df["score"].min() / width) * width)
    hi = int(np.ceil((df["score"].max() + 1) / width) * width)
    edges = list(range(lo, hi + width, width))
    band = pd.cut(df["score"], edges, right=False)
    g = df.groupby(band, observed=True).agg(
        loans=("loan_id", "count"), bads=("bad", "sum"),
        amount=("amount_financed", "sum"), net_loss=("net_loss", "sum"),
        est_margin=("est_margin", "sum"), est_profit=("est_profit", "sum"),
    )
    g["bad_rate"] = g["bads"] / g["loans"]
    g["net_loss_rate"] = g["net_loss"] / g["amount"]
    g["margin_rate"] = g["est_margin"] / g["amount"]
    g.index = g.index.astype(str)
    return g.reset_index(names="score_band")


def cutoff_table(df: pd.DataFrame, cutoffs: range | None = None) -> pd.DataFrame:
    """Approval rate, approved bad rate, losses avoided and profit at each cutoff."""
    cutoffs = cutoffs or range(int(df["score"].quantile(0.0) // 10 * 10), int(df["score"].quantile(0.6)), 10)
    base_profit = df["est_profit"].sum()
    rows = []
    for c in cutoffs:
        ap = df[df["score"] >= c]
        dec = df[df["score"] < c]
        rows.append({
            "cutoff": c,
            "approval_rate": len(ap) / len(df),
            "approved_bad_rate": ap["bad"].mean() if len(ap) else np.nan,
            "declined_bad_rate": dec["bad"].mean() if len(dec) else np.nan,
            "approved_net_loss_rate": ap["net_loss"].sum() / ap["amount_financed"].sum() if len(ap) else np.nan,
            "net_loss_avoided": dec["net_loss"].sum(),
            "margin_forgone": dec["est_margin"].sum(),
            "profit_change": ap["est_profit"].sum() - base_profit,
        })
    return pd.DataFrame(rows)


def recommend_cutoff(table: pd.DataFrame, min_approval: float = 0.85) -> pd.Series:
    """Highest-profit cutoff that still approves at least ``min_approval`` of volume."""
    eligible = table[table["approval_rate"] >= min_approval]
    return eligible.loc[eligible["profit_change"].idxmax()]
