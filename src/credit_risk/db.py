"""DuckDB helpers: load the portfolio tables and run the SQL files in ``sql/``."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import CONFIG, Config

SEGMENTS = ("product", "new_used", "fico_band", "ltv_band", "term_band")


def connect(cfg: Config = CONFIG, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(cfg.db_path), read_only=read_only)


def load_tables(con: duckdb.DuckDBPyConnection, tables: dict[str, pd.DataFrame]) -> None:
    """(Re)create each DataFrame as a DuckDB table."""
    for name, df in tables.items():
        con.register("_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")


def run_sql(con: duckdb.DuckDBPyConnection, name: str, cfg: Config = CONFIG,
            params: dict | None = None, **fmt) -> pd.DataFrame:
    """Run ``sql/<name>.sql`` with optional named parameters and returns a DataFrame."""
    sql = Path(cfg.sql_dir, f"{name}.sql").read_text()
    if fmt:
        sql = sql.format(**fmt)
    return con.execute(sql, params or {}).df()


def segmentation(con, segment: str, as_of: str, cfg: Config = CONFIG) -> pd.DataFrame:
    if segment not in SEGMENTS:  # whitelist: the column name is formatted into the SQL
        raise ValueError(f"segment must be one of {SEGMENTS}")
    iso = pd.Timestamp(as_of).date().isoformat()
    return run_sql(con, "segmentation", cfg, params={"as_of": iso}, segment=segment)
