import numpy as np

from credit_risk.generate_data import CHARGED_OFF, PAID_OFF, build_portfolio


def test_reproducible(cfg, portfolio):
    again = build_portfolio(cfg)
    assert again["loans"].equals(portfolio["loans"])
    assert again["performance"]["balance"].sum() == portfolio["performance"]["balance"].sum()


def test_loan_attributes_in_range(portfolio):
    loans = portfolio["loans"]
    assert loans["loan_id"].is_unique
    assert loans["fico"].between(540, 840).all()
    assert loans["term_months"].isin([36, 48, 60, 72]).all()
    assert (loans["amount_financed"] > 0).all()
    assert loans["apr"].between(3.99, 24.99).all()


def test_terminal_states_are_final(portfolio):
    perf = portfolio["performance"]
    terminal = perf[perf["status"].isin([CHARGED_OFF, PAID_OFF])]
    # each loan reaches a terminal state at most once, and it is its last observation
    assert terminal["loan_id"].is_unique
    last_month = perf.groupby("loan_id")["month"].max()
    assert (terminal.set_index("loan_id")["month"] == last_month.loc[terminal["loan_id"]]).all()


def test_chargeoff_accounting(portfolio):
    perf = portfolio["performance"]
    co = perf[perf["status"] == CHARGED_OFF]
    assert (co["chargeoff_amt"] > 0).all()
    assert (co["recovery_amt"] <= co["chargeoff_amt"]).all()
    assert (perf.loc[perf["status"] != CHARGED_OFF, "chargeoff_amt"] == 0).all()
    assert (perf.loc[perf["status"].isin([CHARGED_OFF, PAID_OFF]), "balance"] == 0).all()


def test_loss_levels_are_plausible(portfolio):
    loans, perf = portfolio["loans"], portfolio["performance"]
    net = (perf["chargeoff_amt"] - perf["recovery_amt"]).sum() / loans["amount_financed"].sum()
    assert 0.01 < net < 0.10
    assert np.isfinite(perf["balance"]).all()
