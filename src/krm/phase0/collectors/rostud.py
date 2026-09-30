"""Rostrud bulk CSV/XLSX collector for offline datasets from data-in.ru.

Reads CSV and XLSX files from a configured directory, maps each row through
schema.map_rostud(), normalises, and stores in the Phase 0 DuckDB database.

Chunked reading (chunksize=10 000) avoids OOM on multi-million-record files.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from krm.phase0.schema import enrich_record, map_rostud, normalize_record
from krm.phase0.storage import (
    finish_phase0_run,
    get_phase0_connection,
    init_phase0_tables,
    start_phase0_run,
    upsert_phase0_vacancy,
)

_CHUNK_SIZE = 10_000

logger = logging.getLogger(__name__)


def collect_rostud(config: Any = None) -> int:
    """Scan the Rostrud dataset directory for CSV/XLSX files, map each row
    to the hh.ru schema, and store in the Phase 0 DuckDB database.

    Returns the total number of records successfully stored.
    """
    if config is None:
        from krm.config import Config

        config = Config()

    if not config.phase0_rostud_enabled:
        print("Rostrud collector disabled")
        return 0

    dataset_path = Path(config.phase0_rostud_dataset_path)
    if not dataset_path.is_dir():
        logger.warning("Rostrud dataset path not found: %s", dataset_path)
        return 0

    conn = get_phase0_connection(config.phase0_db_path)
    init_phase0_tables(conn)

    total_stored = 0

    for file_path in sorted(dataset_path.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls"):
            continue

        run_id = f"rostud-{file_path.stem}"
        start_phase0_run(conn, run_id, source="rostud", url_pattern=str(file_path))

        try:
            records_fetched, records_stored = _process_file(
                conn, file_path, run_id
            )
        except Exception:
            logger.exception("Failed to process %s", file_path)
            # Mark run as finished with partial counts
            finish_phase0_run(conn, run_id, records_fetched=0, records_stored=0)
            continue

        finish_phase0_run(conn, run_id, records_fetched, records_stored)
        total_stored += records_stored

    conn.close()
    return total_stored


def _process_file(
    conn: Any, file_path: Path, run_id: str
) -> tuple[int, int]:
    """Process a single CSV or XLSX file, yielding (fetched, stored) counts."""
    suffix = file_path.suffix.lower()
    records_fetched = 0
    records_stored = 0

    if suffix == ".csv":
        reader = pd.read_csv(file_path, chunksize=_CHUNK_SIZE)
        for chunk in reader:
            fetched, stored = _process_chunk(conn, chunk, run_id)
            records_fetched += fetched
            records_stored += stored
    else:
        df = pd.read_excel(file_path)
        records_fetched, records_stored = _process_chunk(conn, df, run_id)

    return records_fetched, records_stored


def _process_chunk(
    conn: Any, chunk: pd.DataFrame, run_id: str
) -> tuple[int, int]:
    """Process a single DataFrame chunk — map, normalize, upsert.

    Returns (records_fetched, records_stored).
    """
    rows = chunk.to_dict("records")
    records_fetched = 0
    records_stored = 0

    for row in rows:
        _replace_nan_with_none(row)
        records_fetched += 1
        try:
            mapped = map_rostud(row, capture_ts="rostud")
            normalized = normalize_record(mapped)
            enriched = enrich_record(normalized)
            vacancy_id = enriched["id"]
            upsert_phase0_vacancy(conn, run_id, vacancy_id, enriched)
            records_stored += 1
        except ValueError:
            continue

    return records_fetched, records_stored


def _replace_nan_with_none(row: dict[str, Any]) -> None:
    """Mutate the dict in-place: replace NaN and pd.NA with None."""
    for key, value in list(row.items()):
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            row[key] = None
        elif value is pd.NA:
            row[key] = None
