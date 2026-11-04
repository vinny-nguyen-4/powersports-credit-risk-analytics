"""Synthetic powersports installment-loan portfolio.

Builds three tables with a fixed random seed:

* ``macro``        – monthly unemployment index that drives delinquency pressure
* ``loans``        – one row per originated loan (application + collateral attributes)
* ``performance``  – one row per loan per month on book (status, balance, charge-offs)

The data is fully synthetic. It is designed to behave like a real consumer
powersports book (seasoning curves, macro sensitivity, risk driven by FICO /
DTI / LTV / term / collateral) so the downstream analytics are realistic,
but no real customer or company data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, Config

PRODUCTS = ["motorcycle", "atv", "side_by_side", "watercraft"]
PRODUCT_MIX = [0.45, 0.22, 0.21, 0.12]
PRODUCT_PRICE = {"motorcycle": 14_000, "atv": 10_000, "side_by_side": 20_000, "watercraft": 24_000}
PRODUCT_RISK = {"motorcycle": 0.15, "atv": 0.10, "side_by_side": 0.0, "watercraft": -0.10}
PRODUCT_RECOVERY = {"motorcycle": 0.45, "atv": 0.40, "side_by_side": 0.50, "watercraft": 0.35}

# Delinquency status codes used in ``performance.status``
CURRENT, DPD30, DPD60, DPD90, CHARGED_OFF, PAID_OFF = range(6)
STATUS_LABELS = {CURRENT: "current", DPD30: "30dpd", DPD60: "60dpd", DPD90: "90+dpd",
                 CHARGED_OFF: "charged_off", PAID_OFF: "paid_off"}


def month_range(start: str, end: str) -> pd.PeriodIndex:
    return pd.period_range(start, end, freq="M")


def generate_macro(cfg: Config = CONFIG, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Unemployment path: low in 2022, a 2024 softening, partial recovery after."""
    rng = rng or np.random.default_rng(cfg.seed)
    months = month_range(cfg.first_orig_month, cfg.as_of_month)
    anchors_x = [0, 12, 26, 32, 44, len(months) - 1]
    anchors_y = [3.7, 3.6, 4.4, 4.6, 4.2, 4.3]
    path = np.interp(np.arange(len(months)), anchors_x, anchors_y)
    path = path + rng.normal(0, 0.04, len(months))
    return pd.DataFrame({"month": months.to_timestamp(), "unemployment_rate": path.round(2)})


def _amortized_payment(principal: np.ndarray, apr: np.ndarray, term: np.ndarray) -> np.ndarray:
    r = apr / 100 / 12
    return principal * r / (1 - (1 + r) ** (-term))


def _scheduled_balance(principal, apr, term, payments_made):
    r = apr / 100 / 12
    pmt = _amortized_payment(principal, apr, term)
    k = np.minimum(payments_made, term)
    bal = principal * (1 + r) ** k - pmt * ((1 + r) ** k - 1) / r
    return np.clip(bal, 0, None)


def generate_loans(cfg: Config = CONFIG, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(cfg.seed)
    n = cfg.n_loans
    months = month_range(cfg.first_orig_month, cfg.last_orig_month)

    # Spring/summer riding-season peak plus modest volume growth
    idx = np.arange(len(months))
    season = 1 + 0.35 * np.sin(2 * np.pi * (months.month.values - 2) / 12)
    weights = season * (1 + 0.01 * idx)
    orig_idx = rng.choice(len(months), size=n, p=weights / weights.sum())

    product = rng.choice(PRODUCTS, size=n, p=PRODUCT_MIX)
    is_new = rng.random(n) < 0.70
    fico = np.clip(rng.normal(705, 58, n), 540, 840).round().astype(int)
    income = np.exp(rng.normal(np.log(68_000), 0.45, n)).round(-2)
    base_price = np.array([PRODUCT_PRICE[p] for p in product])
    price = base_price * np.exp(rng.normal(0, 0.35, n)) * np.where(is_new, 1.0, 0.72)

    # Better credit tends to put more money down
    down_pct = np.clip(rng.beta(2, 12, n) + 0.0015 * (fico - 700), 0.0, 0.5)
    backend = price * rng.uniform(0.02, 0.12, n)  # warranty, GAP, fees
    amount = (price * (1 - down_pct) + backend).round(2)
    ltv = amount / price

    term_choices = np.array([36, 48, 60, 72])
    term_p = np.where(amount[:, None] > 15_000, [0.08, 0.17, 0.40, 0.35], [0.20, 0.30, 0.35, 0.15])
    term = np.array([rng.choice(term_choices, p=p) for p in term_p])

    apr = np.clip(4.99 + 0.055 * (780 - fico) + 0.6 * (term == 72) + rng.normal(0, 0.8, n), 3.99, 24.99).round(2)
    dti = np.clip(rng.beta(3, 6, n) * 0.7, 0.03, 0.60).round(3)
    payment = _amortized_payment(amount, apr, term).round(2)
    pti = (payment / (income / 12)).round(4)

    # Latent risk multiplier (unobserved noise keeps the scorecard honest)
    log_risk = (
        -0.75 * (fico - 700) / 50
        + 2.0 * (dti - 0.30)
        + 1.6 * (ltv - 1.0)
        + 6.0 * (pti - 0.060)
        + 0.18 * (term == 72)
        + 0.22 * (~is_new)
        + np.array([PRODUCT_RISK[p] for p in product])
        + rng.normal(0, 0.75, n)
    )

    loans = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "orig_month": months[orig_idx].to_timestamp(),
        "product": product,
        "new_used": np.where(is_new, "new", "used"),
        "fico": fico,
        "annual_income": income,
        "vehicle_price": price.round(2),
        "down_payment_pct": down_pct.round(3),
        "amount_financed": amount,
        "ltv": ltv.round(3),
        "term_months": term,
        "apr": apr,
        "dti": dti,
        "monthly_payment": payment,
        "pti": pti,
        "_risk": np.exp(log_risk),
        "_orig_idx": orig_idx,
    })
    return loans.sort_values(["orig_month", "loan_id"]).reset_index(drop=True)


def simulate_performance(loans: pd.DataFrame, macro: pd.DataFrame, cfg: Config = CONFIG,
                         rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Monthly Markov simulation of delinquency states for every loan."""
    rng = rng or np.random.default_rng(cfg.seed + 1)
    cal = month_range(cfg.first_orig_month, cfg.as_of_month)
    n = len(loans)
    risk = loans["_risk"].to_numpy()
    orig_idx = loans["_orig_idx"].to_numpy()
    principal = loans["amount_financed"].to_numpy()
    apr = loans["apr"].to_numpy()
    term = loans["term_months"].to_numpy()
    rec_mean = loans["product"].map(PRODUCT_RECOVERY).to_numpy() - 0.06 * (loans["new_used"] == "used").to_numpy()
    unemp = macro["unemployment_rate"].to_numpy()

    state = np.full(n, CURRENT)
    paid = np.zeros(n, dtype=int)  # payments made
    frames = []

    for t in range(len(cal)):
        active = (orig_idx < t) & (state < CHARGED_OFF)  # first payment due the month after origination
        if not active.any():
            continue
        age = t - orig_idx
        macro_mult = np.exp(0.55 * (unemp[t] - 4.0))
        u = rng.random(n)
        new_state = state.copy()
        chargeoff = np.zeros(n)
        recovery = np.zeros(n)

        # current -> 30 / prepay / pay
        seasoning = 0.0014 + 0.0050 * (age / 12) * np.exp(1 - age / 12)
        p_30 = np.clip(seasoning * risk * macro_mult, 0, 0.5)
        cur = active & (state == CURRENT)
        to30 = cur & (u < p_30)
        prepay = cur & ~to30 & (u > 1 - 0.014)
        new_state[to30] = DPD30
        paid[cur & ~to30] += 1
        new_state[prepay] = PAID_OFF

        # roll / cure from delinquent buckets (worse risk rolls more)
        roll_adj = np.clip(risk ** 0.25, 0.6, 1.8)
        for bucket, p_cure, p_roll in ((DPD30, 0.45, 0.30), (DPD60, 0.22, 0.55), (DPD90, 0.08, 0.72)):
            m = active & (state == bucket)
            pr = np.clip(p_roll * roll_adj * macro_mult ** 0.5, 0, 0.95)
            pc = p_cure / roll_adj
            cure = m & (u < pc)
            roll = m & ~cure & (u > 1 - pr)
            new_state[cure] = CURRENT
            paid[cure] = age[cure]  # brought fully current
            new_state[roll] = bucket + 1  # 90+ rolls to charge-off at 120 DPD

        # balances and charge-off losses
        balance = _scheduled_balance(principal, apr, term, paid)
        co = active & (new_state == CHARGED_OFF)
        chargeoff[co] = balance[co]
        rec_rate = np.clip(rng.normal(rec_mean, 0.12), 0.05, 0.85)
        recovery[co] = chargeoff[co] * rec_rate[co]

        matured = active & (new_state == CURRENT) & (paid >= term)
        new_state[matured] = PAID_OFF
        balance[(new_state == PAID_OFF) | co] = 0.0

        frames.append(pd.DataFrame({
            "loan_id": loans["loan_id"].to_numpy()[active],
            "month": cal[t].to_timestamp(),
            "mob": age[active],
            "status": new_state[active],
            "balance": balance[active].round(2),
            "chargeoff_amt": chargeoff[active].round(2),
            "recovery_amt": recovery[active].round(2),
        }))
        state = new_state

    perf = pd.concat(frames, ignore_index=True)
    perf["status_label"] = perf["status"].map(STATUS_LABELS)
    return perf


def build_portfolio(cfg: Config = CONFIG) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    macro = generate_macro(cfg, rng)
    loans = generate_loans(cfg, rng)
    perf = simulate_performance(loans, macro, cfg)
    public_loans = loans.drop(columns=["_risk", "_orig_idx"])
    return {"macro": macro, "loans": public_loans, "performance": perf}
