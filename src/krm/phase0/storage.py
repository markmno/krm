"""Phase 0 DuckDB/Parquet storage layer.

Phase 0 uses its own DuckDB database for raw vacancy collection from historical
sources (Wayback CDX, Census API, trudvsem, rostud). This module provides the
connection, table init, and upsert helpers matching the existing pipeline schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# --- Connection ----------------------------------------------------------------


def get_phase0_connection(db_path: str | None = None) -> Any:
    """Return a DuckDB connection for the Phase 0 database.

    Args:
        db_path: Path to the DuckDB database. If None, reads from config.

    Returns:
        A DuckDB connection object.
    """
    import duckdb

    if db_path is None:
        from src.krm.config import Config

        config = Config()
        db_path = config.phase0_db_path

    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(p))


# --- Table initialization -----------------------------------------------------


def init_phase0_tables(conn: Any) -> None:
    """Create Phase 0 DuckDB tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_vacancies (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            data JSON NOT NULL,
            fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS phase0_runs (
            run_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_url_pattern TEXT,
            started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            records_fetched INTEGER DEFAULT 0,
            records_stored INTEGER DEFAULT 0,
            params JSON
        )
    """)


# --- Raw vacancy upsert -------------------------------------------------------


def upsert_phase0_vacancy(
    conn: Any, run_id: str, vacancy_id: str, data: dict[str, Any]
) -> None:
    """Insert or ignore a raw vacancy record.

    If a vacancy with the same id already exists (from a previous run),
    the insert is silently ignored.

    Args:
        conn: DuckDB connection.
        run_id: The Phase 0 run identifier.
        vacancy_id: Unique vacancy id (e.g., URL or source-specific id).
        data: Vacancy data dict to store as JSON.
    """
    import json

    conn.execute(
        "INSERT OR IGNORE INTO raw_vacancies (id, run_id, data) VALUES (?, ?, ?)",
        [vacancy_id, run_id, json.dumps(data, ensure_ascii=False)],
    )


# --- Run lifecycle ------------------------------------------------------------


def start_phase0_run(
    conn: Any,
    run_id: str,
    source: str,
    url_pattern: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Record the start of a Phase 0 collection run.

    Args:
        conn: DuckDB connection.
        run_id: Unique run identifier.
        source: Source name (e.g., 'hhru_wayback', 'linkedin_wayback', 'trudvsem').
        url_pattern: Optional URL pattern or identifier for the CDX query.
        params: Optional extra parameters stored as JSON.
    """
    import json

    sql = (
        "INSERT INTO phase0_runs"
        + " (run_id, source, source_url_pattern, started_at, params)"
        + " VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)"
    )
    conn.execute(
        sql,
        [
            run_id,
            source,
            url_pattern,
            json.dumps(params) if params is not None else None,
        ],
    )


def finish_phase0_run(
    conn: Any, run_id: str, records_fetched: int, records_stored: int
) -> None:
    """Mark a Phase 0 collection run as completed.

    Args:
        conn: DuckDB connection.
        run_id: Unique run identifier.
        records_fetched: Total records fetched from the source.
        records_stored: Total records successfully stored in raw_vacancies.
    """
    sql = (
        "UPDATE phase0_runs"
        + " SET completed_at = CURRENT_TIMESTAMP,"
        + "    records_fetched = ?,"
        + "    records_stored = ?"
        + " WHERE run_id = ?"
    )
    conn.execute(sql, [records_fetched, records_stored, run_id])


# --- Merge into main pipeline -------------------------------------------------


def merge_into_main(
    phase0_db_path: str, main_db_path: str | None = None
) -> int:
    """Merge Phase 0 raw_vacancies into the main pipeline DuckDB database.

    Uses INSERT OR IGNORE so existing main-pipeline vacancies are not
    overwritten by Phase 0 data.

    Args:
        phase0_db_path: Path to the Phase 0 DuckDB database.
        main_db_path: Path to the main pipeline DuckDB database.
            Defaults to data/krm.duckdb.

    Returns:
        Number of rows merged into the main database.
    """
    import duckdb

    if main_db_path is None:
        main_db_path = str(Path("data/krm.duckdb"))

    # Ensure the main DB directory exists
    Path(main_db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    try:
        _ = conn.execute(f"ATTACH '{phase0_db_path}' AS phase0_db (READ_ONLY)")
        _ = conn.execute(f"ATTACH '{main_db_path}' AS main_db")

        # Ensure main_db has the raw_vacancies table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS main_db.raw_vacancies (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                data JSON NOT NULL,
                fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        result = conn.execute(
            "INSERT OR IGNORE INTO main_db.raw_vacancies"
            + " SELECT * FROM phase0_db.raw_vacancies"
        )

        return result.fetchall()[0][0]
    finally:
        conn.close()
