import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk import db  # noqa: E402
from credit_risk.config import CONFIG  # noqa: E402
from credit_risk.generate_data import build_portfolio  # noqa: E402


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    return replace(CONFIG, n_loans=6000, data_dir=tmp / "data", reports_dir=tmp / "reports")


@pytest.fixture(scope="session")
def portfolio(cfg):
    return build_portfolio(cfg)


@pytest.fixture(scope="session")
def con(cfg, portfolio):
    c = db.connect(cfg)
    db.load_tables(c, portfolio)
    yield c
    c.close()
