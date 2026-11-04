"""Application scorecard: WoE binning -> logistic regression -> points scaling.

Target: ``bad = 1`` if the loan reached 60+ days past due or charged off within
``bad_window_mob`` months on book. Only vintages that are fully seasoned to that
window are used, split into a development sample and an out-of-time (OOT) sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import CONFIG, Config

NUMERIC_FEATURES = ["fico", "dti", "ltv", "pti", "down_payment_pct", "term_months"]
CATEGORICAL_FEATURES = ["product", "new_used"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_target(loans: pd.DataFrame, perf: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Attach the ``bad`` flag and a ``sample`` label (dev / oot) to seasoned loans."""
    window = perf[perf["mob"] <= cfg.bad_window_mob]
    bad_ids = window.loc[window["status"].isin([2, 3, 4]), "loan_id"].unique()
    seasoned_cutoff = pd.Timestamp(cfg.as_of_month) - pd.DateOffset(months=cfg.bad_window_mob)
    df = loans[loans["orig_month"] <= min(seasoned_cutoff, pd.Timestamp(cfg.oot_end_month))].copy()
    df["bad"] = df["loan_id"].isin(bad_ids).astype(int)
    df["sample"] = np.where(df["orig_month"] <= pd.Timestamp(cfg.dev_end_month), "dev", "oot")
    return df


@dataclass
class BinTable:
    feature: str
    edges: list[float] | None   # numeric bin edges (None for categorical)
    table: pd.DataFrame         # bin, count, bads, goods, bad_rate, woe, iv

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())


def _woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"bin": bins.astype(str), "bad": y}).groupby("bin", observed=True)["bad"].agg(["count", "sum"])
    g.columns = ["count", "bads"]
    g["goods"] = g["count"] - g["bads"]
    # +0.5 smoothing avoids infinite WoE in empty cells
    dist_good = (g["goods"] + 0.5) / (g["goods"].sum() + 0.5 * len(g))
    dist_bad = (g["bads"] + 0.5) / (g["bads"].sum() + 0.5 * len(g))
    g["bad_rate"] = g["bads"] / g["count"]
    g["woe"] = np.log(dist_good / dist_bad)
    g["iv"] = (dist_good - dist_bad) * g["woe"]
    return g.reset_index()


def fit_bins(df: pd.DataFrame, feature: str, n_bins: int) -> BinTable:
    if feature in CATEGORICAL_FEATURES:
        return BinTable(feature, None, _woe_table(df[feature], df["bad"]))
    qs = np.unique(np.quantile(df[feature], np.linspace(0, 1, n_bins + 1)))
    edges = [-np.inf, *qs[1:-1], np.inf]
    bins = pd.cut(df[feature], edges, right=False)
    return BinTable(feature, edges, _woe_table(bins, df["bad"]))


def apply_bins(df: pd.DataFrame, bt: BinTable) -> pd.Series:
    """Map raw values to their bin's WoE (unseen categories get 0 = neutral)."""
    if bt.edges is None:
        bins = df[bt.feature].astype(str)
    else:
        bins = pd.cut(df[bt.feature], bt.edges, right=False).astype(str)
    return bins.map(dict(zip(bt.table["bin"], bt.table["woe"]))).fillna(0.0).astype(float)


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    """Max separation between cumulative bad and good distributions."""
    order = np.argsort(score)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index of ``actual`` vs ``expected`` score distributions."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class Scorecard:
    bins: dict[str, BinTable]
    model: LogisticRegression
    features: list[str]
    factor: float
    offset: float

    def woe_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: apply_bins(df, self.bins[f]) for f in self.features}, index=df.index)

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.woe_matrix(df))[:, 1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Score = offset + factor * ln(good:bad odds); higher = lower risk."""
        p = np.clip(self.predict_pd(df), 1e-6, 1 - 1e-6)
        return np.round(self.offset + self.factor * np.log((1 - p) / p))

    def points_table(self) -> pd.DataFrame:
        """Per-attribute points (they sum to the score, up to rounding)."""
        n = len(self.features)
        b0 = self.model.intercept_[0]
        rows = []
        for f, beta in zip(self.features, self.model.coef_[0]):
            for _, r in self.bins[f].table.iterrows():
                pts = -(beta * r["woe"] + b0 / n) * self.factor + self.offset / n
                rows.append({"feature": f, "bin": r["bin"], "count": int(r["count"]),
                             "bad_rate": r["bad_rate"], "woe": r["woe"], "points": round(pts)})
        return pd.DataFrame(rows)


def train_scorecard(df: pd.DataFrame, cfg: Config = CONFIG, min_iv: float = 0.02) -> tuple[Scorecard, pd.DataFrame]:
    dev = df[df["sample"] == "dev"]
    bins = {f: fit_bins(dev, f, cfg.n_bins) for f in FEATURES}
    iv = pd.DataFrame({"feature": list(bins), "iv": [b.iv for b in bins.values()]}).sort_values("iv", ascending=False)
    selected = iv.loc[iv["iv"] >= min_iv, "feature"].tolist()

    X = pd.DataFrame({f: apply_bins(dev, bins[f]) for f in selected}, index=dev.index)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(X, dev["bad"])
    # Note: with WoE = ln(good/bad), coefficients come out negative for P(bad).
    factor = cfg.pdo / np.log(2)
    offset = cfg.base_score - factor * np.log(cfg.base_odds)
    return Scorecard(bins, model, selected, factor, offset), iv


def evaluate(sc: Scorecard, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in df.groupby("sample"):
        p = sc.predict_pd(part)
        auc = roc_auc_score(part["bad"], p)
        rows.append({"sample": name, "loans": len(part), "bad_rate": part["bad"].mean(),
                     "auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(part["bad"], -p)})
    out = pd.DataFrame(rows).set_index("sample").loc[["dev", "oot"]].reset_index()
    s_dev, s_oot = sc.score(df[df["sample"] == "dev"]), sc.score(df[df["sample"] == "oot"])
    out["psi_vs_dev"] = [0.0, psi(s_dev, s_oot)]
    return out
