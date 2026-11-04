"""Static report charts (PNG) used in the README and the Excel workbook."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots (fixed order, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})
PCT = mtick.PercentFormatter(1.0, decimals=1)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def portfolio_trend(trend: pd.DataFrame, out: Path) -> Path:
    t = trend[trend["month"] >= "2022-07-01"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    panels = [("dq30_plus_rate", "30+ DPD delinquency rate (% of balance)", PCT),
              ("annualized_nco_rate", "Annualized net charge-off rate", PCT),
              ("unemployment_rate", "Unemployment rate (simulated macro driver, %)", None)]
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.plot(t["month"], t[col], color=SERIES[0])
        ax.set_title(title)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        last = t.iloc[-1]
        ax.annotate(f"{last[col]:.1%}" if fmt else f"{last[col]:.1f}", (last["month"], last[col]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    return _save(fig, out)


def vintage_curves(vint: pd.DataFrame, out: Path) -> Path:
    v = vint.assign(year=vint["vintage"].str[:4])
    yearly = (v.groupby(["year", "vintage"]).apply(lambda g: g.set_index("mob")["cum_net_loss_rate"], include_groups=False)
              .groupby(level=["year", "mob"]).agg(["mean", "count"]))
    # only plot MOBs observed for every quarterly vintage in the year, so the average never jumps
    n_q = v.groupby("year")["vintage"].nunique()
    yearly = yearly[yearly["count"] == yearly.index.get_level_values("year").map(n_q)]["mean"].unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, year in enumerate(yearly.columns):
        s = yearly[year].dropna()
        ax.plot(s.index, s.values, color=SERIES[i], label=f"{year} vintages")
        ax.annotate(year, (s.index[-1], s.values[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_title("Cumulative net loss rate by months on book (avg of quarterly vintages)")
    ax.set_xlabel("Months on book")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper left")
    return _save(fig, out)


def score_band_bad_rate(bands: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(bands))
    ax.bar(x, bands["bad_rate"], color=SERIES[0], width=0.8)
    for i, r in bands.iterrows():
        ax.annotate(f"{r['bad_rate']:.0%}", (i, r["bad_rate"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.set_xticks(list(x), [b.replace(".0", "") for b in bands["score_band"]], rotation=0, fontsize=8)
    ax.set_title("Out-of-time bad rate by score band (bad = 60+ DPD within 18 months)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Score band")
    return _save(fig, out)


def cutoff_tradeoff(cut: pd.DataFrame, rec: pd.Series, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(cut["approval_rate"], cut["approved_bad_rate"], color=SERIES[0], marker="o", markersize=5)
    for _, r in cut.iloc[::2].iterrows():
        ax.annotate(f"{int(r['cutoff'])}", (r["approval_rate"], r["approved_bad_rate"]), xytext=(-6, -14),
                    textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.scatter([rec["approval_rate"]], [rec["approved_bad_rate"]], s=110, color=SERIES[1], zorder=5,
               edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"Recommended cutoff {int(rec['cutoff'])}", (rec["approval_rate"], rec["approved_bad_rate"]),
                xytext=(10, 10), textcoords="offset points", fontsize=9, color=INK)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Approval rate (share of booked volume kept)")
    ax.set_ylabel("Bad rate of approved loans")
    ax.set_title("Cutoff trade-off: approval rate vs bad rate (labels = score cutoff)")
    return _save(fig, out)


def forecast_backtest(bt: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = {"actual": "Actual", "vintage_curve": "Vintage-curve forecast", "roll_rate": "Roll-rate forecast"}
    for i, (col, lab) in enumerate(labels.items()):
        ax.plot(bt["month"], bt[col] / 1e3, color=SERIES[i], label=lab, marker="o" if col == "actual" else None,
                markersize=4)
    ax.set_title("Backtest: 12-month net charge-off forecast made Sep 2025 vs actual")
    ax.set_ylabel("Net charge-offs ($K)")
    ax.legend(loc="upper right")
    return _save(fig, out)


def segment_losses(fico: pd.DataFrame, ltv: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, df, title in ((axes[0], fico, "By FICO band"), (axes[1], ltv, "By LTV band")):
        labels = df["segment"].str.split(": ").str[1]
        ax.bar(labels, df["seasoned_net_loss_rate"], color=SERIES[0], width=0.75)
        for i, v in enumerate(df["seasoned_net_loss_rate"]):
            ax.annotate(f"{v:.1%}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        color=INK2)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].yaxis.set_major_formatter(PCT)
    fig.suptitle("Lifetime net loss rate, vintages seasoned 18+ months", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, color=INK)
    return _save(fig, out)
