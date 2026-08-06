"""I/O utilities for KRM pipeline.

All phases exchange data through typed Parquet files. This module provides
consistent read/write helpers with schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa


# --- Parquet helpers -----------------------------------------------------------


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(Path(path))


def write_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Write a DataFrame to Parquet, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    import json

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """Write a list of dicts to a JSONL file."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- DuckDB helpers ------------------------------------------------------------


def get_connection(db_path: Path | str = "data/krm.duckdb") -> Any:
    """Return a DuckDB connection, creating the database if needed."""
    import duckdb

    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(p))


def init_tables(conn: Any) -> None:
    """Create DuckDB tables for raw vacancy storage."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scrape_runs (
            run_id TEXT PRIMARY KEY,
            keyword TEXT NOT NULL,
            query_params JSON,
            started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            vacancies_found INTEGER DEFAULT 0,
            vacancies_fetched INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_vacancies (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            data JSON NOT NULL,
            fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def upsert_raw_vacancy(conn: Any, run_id: str, vacancy_id: str, data: dict[str, Any]) -> None:
    """Insert or ignore a raw vacancy record."""
    import json

    conn.execute(
        "INSERT OR IGNORE INTO raw_vacancies (id, run_id, data) VALUES (?, ?, ?)",
        [vacancy_id, run_id, json.dumps(data, ensure_ascii=False)],
    )


# --- Schema validation ---------------------------------------------------------


def validate_schema(df: pd.DataFrame, schema: dict[str, str]) -> None:
    """Validate that a DataFrame has required columns of expected types."""
    for col, dtype in schema.items():
        if col not in df.columns:
            msg = f"Missing column: {col}"
            raise ValueError(msg)
        actual = str(df[col].dtype)
        if not actual.startswith(dtype):
            msg = f"Column {col}: expected {dtype}, got {actual}"
            raise ValueError(msg)
