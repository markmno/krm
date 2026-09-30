"""Phase 0 data verification module.

Reads raw_vacancies from the Phase 0 DuckDB, checks schema compliance,
field presence, date validity, and duplicate IDs, then produces a
verification_report.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from krm.phase0.schema import OPTIONAL_FIELDS, validate_record


# --- Report type ----------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-?\d{2}-?\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")


# --- Verification ---------------------------------------------------------------


def verify_phase0_data(conn: Any) -> dict[str, Any]:
    """Run all verification checks on Phase 0 raw_vacancies and return a report dict.

    Args:
        conn: A DuckDB connection with the raw_vacancies table populated.

    Returns:
        A dict with keys: summary, schema_compliance, field_presence,
        date_validity, dedup.
    """
    rows = _fetch_all_rows(conn)
    records = _parse_records(rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(records)
    invalid_count = 0
    schema_errors_by_record: dict[str, list[str]] = {}
    for rec in records:
        errs = validate_record(rec)
        if errs:
            invalid_count += 1
            schema_errors_by_record[rec.get("id", "?")] = errs

    summary: dict[str, Any] = {
        "total_records": total,
        "valid_records": total - invalid_count,
        "invalid_records": invalid_count,
    }

    # ── Schema compliance ──────────────────────────────────────────────────────
    schema_compliance: dict[str, Any] = {
        "error_count": invalid_count,
        "errors": [
            {"record_id": rid, "errors": errs}
            for rid, errs in schema_errors_by_record.items()
        ],
    }

    # ── Field presence ─────────────────────────────────────────────────────────
    field_presence: dict[str, dict[str, Any]] = {}
    for field in sorted(OPTIONAL_FIELDS):
        present = sum(1 for r in records if r.get(field) is not None)
        absent = total - present
        field_presence[field] = {
            "present": present,
            "absent": absent,
            "percentage": round(present / total * 100, 2) if total > 0 else 0.0,
        }

    # ── Date validity ──────────────────────────────────────────────────────────
    invalid_dates: list[dict[str, str]] = []
    for rec in records:
        ts = rec.get("_phase0_capture_ts")
        if ts is None or not isinstance(ts, str) or not _ISO_DATE_RE.match(ts):
            invalid_dates.append({
                "record_id": rec.get("id", "?"),
                "capture_ts": repr(ts),
            })

    date_validity: dict[str, Any] = {
        "invalid_capture_ts": len(invalid_dates),
        "details": invalid_dates,
    }

    # ── Dedup ──────────────────────────────────────────────────────────────────
    id_counts: dict[str, int] = {}
    for rec in records:
        rid = rec.get("id", "?")
        id_counts[rid] = id_counts.get(rid, 0) + 1

    dup_ids = {rid: cnt for rid, cnt in id_counts.items() if cnt > 1}
    dedup: dict[str, Any] = {
        "duplicate_ids": list(dup_ids.keys()),
        "duplicate_count": sum(cnt - 1 for cnt in dup_ids.values()),
    }

    return {
        "summary": summary,
        "schema_compliance": schema_compliance,
        "field_presence": field_presence,
        "date_validity": date_validity,
        "dedup": dedup,
    }


def save_verification_report(report: dict[str, Any], path: Path | str) -> None:
    """Write a verification report to a JSON file.

    Args:
        report: The dict returned by verify_phase0_data.
        path: Output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def verify_phase0_db(db_path: str) -> dict[str, Any]:
    """Open a Phase 0 DuckDB, run verification, and return the report.

    Args:
        db_path: Path to the Phase 0 DuckDB file.

    Returns:
        Verification report dict.
    """
    import duckdb

    conn = duckdb.connect(db_path)
    try:
        return verify_phase0_data(conn)
    finally:
        conn.close()


# --- Internal helpers -----------------------------------------------------------


def _fetch_all_rows(conn: Any) -> list[tuple[str, str, str, Any]]:
    """Fetch all rows from raw_vacancies (id, run_id, data, fetched_at)."""
    return conn.execute(
        "SELECT id, run_id, data, fetched_at FROM raw_vacancies ORDER BY id"
    ).fetchall()


def _parse_records(rows: list[tuple[str, str, str, Any]]) -> list[dict[str, Any]]:
    """Parse JSON data from raw rows into record dicts."""
    records: list[dict[str, Any]] = []
    for row in rows:
        data = json.loads(row[2])
        records.append(data)
    return records
