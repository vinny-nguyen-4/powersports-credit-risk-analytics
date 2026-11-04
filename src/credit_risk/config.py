"""Central configuration for the portfolio simulation and analytics pipeline.

Every value can be overridden with an environment variable of the same name
(e.g. ``N_LOANS=5000 python -m credit_risk.pipeline``) so tests and quick runs
can use a smaller book without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default, cast=str):
    value = os.getenv(name)
    return cast(value) if value is not None else default


@dataclass(frozen=True)
class Config:
    # --- simulation ------------------------------------------------------
    seed: int = _env("SEED", 42, int)
    n_loans: int = _env("N_LOANS", 25_000, int)
    first_orig_month: str = "2022-01"
    last_orig_month: str = "2026-06"
    as_of_month: str = "2026-09"  # last observed performance month

    # --- scorecard -------------------------------------------------------
    bad_window_mob: int = 18          # bad = 60+ DPD or charge-off within 18 months on book
    dev_end_month: str = "2024-06"    # development sample: vintages up to this month
    oot_end_month: str = "2025-03"    # out-of-time sample: next 9 vintages (fully seasoned to 18 MOB)
    n_bins: int = 5
    pdo: float = 20.0                 # points to double the odds
    base_score: float = 600.0
    base_odds: float = 10.0           # good:bad odds at the base score

    # --- strategy --------------------------------------------------------
    cost_of_funds_pct: float = 5.0    # annual cost of funds used for the break-even analysis
    servicing_cost_pct: float = 1.0   # annual servicing / operating cost
    avg_life_years: float = 2.0       # average life of a powersports installment loan

    # --- forecast --------------------------------------------------------
    forecast_horizon: int = 12
    backtest_as_of: str = "2025-09"
    roll_rate_lookback: int = 6

    # --- paths -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    reports_dir: Path = field(default_factory=lambda: Path(_env("REPORTS_DIR", str(ROOT / "reports"))))
    sql_dir: Path = ROOT / "sql"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "portfolio.duckdb"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


CONFIG = Config()
