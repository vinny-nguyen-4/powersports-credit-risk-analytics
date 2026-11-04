"""End-to-end pipeline: simulate -> load DuckDB -> SQL analytics -> scorecard ->
strategy -> forecasts -> charts, Excel report and memo.

Run with ``python -m credit_risk.pipeline``.
"""

from __future__ import annotations

import json

import pandas as pd

from . import charts, db, forecast, report, scorecard, strategy
from .config import CONFIG, Config


FEATURE_LABELS = {"fico": "FICO", "down_payment_pct": "down payment %", "ltv": "LTV", "pti": "payment-to-income",
                  "dti": "DTI", "term_months": "term", "product": "collateral type", "new_used": "new/used"}


def _usd(x: float) -> str:
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}K"


def run(cfg: Config = CONFIG) -> dict:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/7] Simulating {cfg.n_loans:,} loans (seed {cfg.seed})")
    from .generate_data import build_portfolio
    tables = build_portfolio(cfg)
    loans, perf = tables["loans"], tables["performance"]

    print("[2/7] Loading DuckDB and running SQL analytics")
    con = db.connect(cfg)
    db.load_tables(con, tables)
    loans.head(500).to_csv(cfg.data_dir / "sample_loans.csv", index=False)
    trend = db.run_sql(con, "delinquency_trend", cfg)
    vint = db.run_sql(con, "vintage_curves", cfg)
    segs = {s: db.segmentation(con, s, cfg.as_of_month, cfg) for s in db.SEGMENTS}

    print("[3/7] Training application scorecard")
    model_df = scorecard.build_target(loans, perf, cfg)
    sc, iv = scorecard.train_scorecard(model_df, cfg)
    metrics = scorecard.evaluate(sc, model_df)
    points = sc.points_table()
    model_df["score"] = sc.score(model_df)

    print("[4/7] Cutoff strategy analysis")
    net_loss = (perf["chargeoff_amt"] - perf["recovery_amt"]).groupby(perf["loan_id"]).sum()
    oot = strategy.loan_economics(model_df[model_df["sample"] == "oot"], net_loss, cfg)
    bands = strategy.score_bands(oot)
    cuts = strategy.cutoff_table(oot)
    rec = strategy.recommend_cutoff(cuts)

    print("[5/7] Loss forecasts and backtest")
    vc = forecast.vintage_curve_forecast(loans, perf, cfg.as_of_month, cfg.forecast_horizon)
    rr = forecast.roll_rate_forecast(con, loans, perf, cfg.as_of_month, cfg, cfg.forecast_horizon)
    bt = forecast.backtest(con, loans, perf, cfg)
    con.close()
    fc = pd.DataFrame({"vintage_curve": vc["monthly"], "roll_rate": rr["monthly"]}).rename_axis("month").reset_index()

    print("[6/7] Charts")
    f = cfg.figures_dir
    imgs = [
        charts.portfolio_trend(trend, f / "portfolio_trend.png"),
        charts.vintage_curves(vint, f / "vintage_curves.png"),
        charts.segment_losses(segs["fico_band"], segs["ltv_band"], f / "segment_losses.png"),
        charts.score_band_bad_rate(bands, f / "score_band_bad_rate.png"),
        charts.cutoff_tradeoff(cuts, rec, f / "cutoff_tradeoff.png"),
        charts.forecast_backtest(bt, f / "forecast_backtest.png"),
    ]

    print("[7/7] Excel report and memo")
    last = trend.iloc[-1]
    nco_3m = trend["annualized_nco_rate"].iloc[-3:].mean()
    oot_m = metrics.set_index("sample").loc["oot"]
    bt_tot = bt[["actual", "vintage_curve", "roll_rate"]].sum()
    seg_rate = {s: d.set_index("segment")["seasoned_net_loss_rate"] for s, d in segs.items()}
    summary = {
        "as_of": cfg.as_of_month,
        "active_accounts": int(last["active_accounts"]),
        "balance": _usd(last["ending_balance"]),
        "dq30": f"{last['dq30_plus_rate']:.2%}",
        "nco": f"{nco_3m:.2%}",
        "loss_fico_low": f"{seg_rate['fico_band'].iloc[0]:.1%}",
        "loss_fico_high": f"{seg_rate['fico_band'].iloc[-1]:.1%}",
        "loss_ltv_low": f"{seg_rate['ltv_band'].iloc[0]:.1%}",
        "loss_ltv_high": f"{seg_rate['ltv_band'].iloc[-1]:.1%}",
        "loss_36": f"{seg_rate['term_band']['36 mo']:.1%}",
        "loss_72": f"{seg_rate['term_band']['72 mo']:.1%}",
        "oot_auc": float(oot_m["auc"]), "oot_gini": float(oot_m["gini"]), "oot_ks": float(oot_m["ks"]),
        "psi": float(oot_m["psi_vs_dev"]),
        "features": ", ".join(FEATURE_LABELS.get(f, f) for f in sc.features),
        "cutoff": int(rec["cutoff"]),
        "approval": f"{rec['approval_rate']:.1%}",
        "bad_all": f"{oot['bad'].mean():.1%}",
        "bad_cut": f"{rec['approved_bad_rate']:.1%}",
        "loss_avoided": _usd(rec["net_loss_avoided"]),
        "margin_forgone": _usd(rec["margin_forgone"]),
        "profit_lift": _usd(rec["profit_change"]),
        "fc_vintage": _usd(vc["total"]),
        "fc_roll": _usd(rr["total"]),
        "bt_as_of": cfg.backtest_as_of,
        "bt_actual": _usd(bt_tot["actual"]),
        "bt_err_vc": f"{bt_tot['vintage_curve'] / bt_tot['actual'] - 1:+.1%}",
        "bt_err_rr": f"{bt_tot['roll_rate'] / bt_tot['actual'] - 1:+.1%}",
    }
    kpis = [
        ("As-of month", cfg.as_of_month), ("Active accounts", f"{summary['active_accounts']:,}"),
        ("Outstanding balance", summary["balance"]), ("30+ DPD rate (balance)", summary["dq30"]),
        ("Annualized NCO rate (3-mo avg)", summary["nco"]),
        ("Scorecard OOT AUC / KS", f"{summary['oot_auc']:.3f} / {summary['oot_ks']:.3f}"),
        ("Recommended score cutoff", str(summary["cutoff"])),
        ("Approval rate at cutoff", summary["approval"]),
        ("Next-12-mo NCO forecast (vintage curve)", summary["fc_vintage"]),
        ("Next-12-mo NCO forecast (roll rate)", summary["fc_roll"]),
        (f"Backtest error from {cfg.backtest_as_of} (vintage / roll)", f"{summary['bt_err_vc']} / {summary['bt_err_rr']}"),
    ]
    roll = rr["matrix"].reset_index(names="from_bucket")
    seg_all = pd.concat([d.assign(dimension=s) for s, d in segs.items()])[
        ["dimension", "segment", "loans_originated", "amount_originated", "current_balance",
         "dq30_plus_rate", "seasoned_net_loss_rate"]]
    vint_pivot = vint.pivot(index="mob", columns="vintage", values="cum_net_loss_rate").reset_index()
    vint_pivot.columns = ["mob"] + [f"{c}_loss_rate" for c in vint_pivot.columns[1:]]
    sheets = {
        "Delinquency Trend": (trend, "Monthly balances, delinquency and net charge-offs (SQL: delinquency_trend.sql)"),
        "Vintage Curves": (vint_pivot, "Cumulative net loss rate by quarterly vintage and MOB (SQL: vintage_curves.sql)"),
        "Roll Rates": (roll, f"Dollar-weighted transitions, last {cfg.roll_rate_lookback} months (SQL: roll_rates.sql)"),
        "Segmentation": (seg_all, f"Snapshot {cfg.as_of_month}; loss rate on vintages seasoned 18+ months"),
        "Scorecard Points": (points, "Points per attribute; total score = sum of points"),
        "Scorecard IV": (iv, "Information value on development sample (features with IV >= 0.02 kept)"),
        "Model Performance": (metrics, "Dev vs out-of-time discrimination and stability"),
        "Score Bands": (bands, "OOT performance and estimated economics by score band"),
        "Cutoff Strategy": (cuts, "Approval / bad rate / profit trade-off at each cutoff (OOT)"),
        "Loss Forecast": (fc, "Next 12 months net charge-offs, existing book"),
        "Forecast Backtest": (bt, f"Forecast made as of {cfg.backtest_as_of} vs realized"),
    }
    report.build_workbook(cfg.reports_dir / "credit_risk_report.xlsx", sheets, kpis, imgs)
    report.write_memo(cfg.reports_dir / "risk_memo.md", summary)
    (cfg.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
