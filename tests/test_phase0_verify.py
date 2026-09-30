"""Tests for phase0/verify.py — Phase 0 data verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb
import pytest

from krm.phase0.schema import OPTIONAL_FIELDS
from krm.phase0.verify import (
    save_verification_report,
    verify_phase0_data,
    verify_phase0_db,
)


# --- Fixtures ------------------------------------------------------------------


def _make_valid_record(rid: str, source: str = "wayback-hhru") -> dict:
    """Return a minimal valid record with all required fields."""
    return {
        "id": rid,
        "name": f"Vacancy {rid}",
        "_phase0_source": source,
        "_phase0_capture_ts": "2024-01-15T12:00:00",
        "_phase0_original_url": f"https://hh.ru/vacancy/{rid}",
    }


def _make_record_with_optional(rid: str, **kwargs: object) -> dict:
    """Return a valid record with extra optional fields."""
    rec = _make_valid_record(rid)
    rec.update(kwargs)  # type: ignore[arg-type]
    return rec


def _populate_db(conn: duckdb.DuckDBPyConnection, records: list[dict]) -> None:
    """Populate a DuckDB connection with records in raw_vacancies format."""
    import json as _json

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_vacancies (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            data JSON NOT NULL,
            fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for rec in records:
        conn.execute(
            "INSERT OR IGNORE INTO raw_vacancies (id, run_id, data) VALUES (?, ?, ?)",
            [rec["id"], "test-run", _json.dumps(rec, ensure_ascii=False)],
        )


def _make_in_memory_conn_with(records: list[dict]) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection and populate with records."""
    conn = duckdb.connect()
    _populate_db(conn, records)
    return conn


# --- Tests: verify_phase0_data -------------------------------------------------


class TestVerifyPhase0Data:
    """Core verification checks."""

    def test_all_valid_records_produces_clean_report(self) -> None:
        """Given a DB with all valid records,
        When verify_phase0_data is called,
        Then summary shows zero invalid, no schema errors, no date issues, no dups."""
        records = [
            _make_valid_record("v-1"),
            _make_valid_record("v-2", source="wayback-linkedin"),
            _make_valid_record("v-3", source="trudvsem"),
            _make_record_with_optional(
                "v-4",
                description="Test desc",
                employer={"name": "Acme"},
                area={"name": "Moscow"},
            ),
        ]
        conn = _make_in_memory_conn_with(records)
        try:
            report = verify_phase0_data(conn)

            assert report["summary"]["total_records"] == 4
            assert report["summary"]["valid_records"] == 4
            assert report["summary"]["invalid_records"] == 0

            assert report["schema_compliance"]["error_count"] == 0
            assert report["schema_compliance"]["errors"] == []

            assert report["date_validity"]["invalid_capture_ts"] == 0
            assert report["date_validity"]["details"] == []

            assert report["dedup"]["duplicate_ids"] == []
            assert report["dedup"]["duplicate_count"] == 0
        finally:
            conn.close()

    def test_schema_compliance_reports_invalid_records(self) -> None:
        """Given a DB with records that fail validate_record,
        When verify_phase0_data is called,
        Then schema_compliance lists the errors with record ids."""
        records = [
            # Missing name
            {
                "id": "v-bad1",
                "_phase0_source": "wayback-hhru",
                "_phase0_capture_ts": "2024-01-01",
                "_phase0_original_url": "https://hh.ru/vacancy/1",
            },
            # Bad source
            {
                "id": "v-bad2",
                "name": "Bad Source",
                "_phase0_source": "not-a-source",
                "_phase0_capture_ts": "2024-01-01",
                "_phase0_original_url": "https://hh.ru/vacancy/2",
            },
            # Missing original_url
            {
                "id": "v-bad3",
                "name": "No URL",
                "_phase0_source": "rostud",
                "_phase0_capture_ts": "2024-01-01",
                "_phase0_original_url": None,
            },
            # Valid record mixed in
            _make_valid_record("v-ok"),
        ]
        conn = _make_in_memory_conn_with(records)
        try:
            report = verify_phase0_data(conn)

            assert report["summary"]["total_records"] == 4
            assert report["summary"]["invalid_records"] == 3

            assert report["schema_compliance"]["error_count"] == 3
            error_ids = {e["record_id"] for e in report["schema_compliance"]["errors"]}
            assert error_ids == {"v-bad1", "v-bad2", "v-bad3"}

            # Check specific error messages
            for err_entry in report["schema_compliance"]["errors"]:
                if err_entry["record_id"] == "v-bad1":
                    assert any("name" in e.lower() for e in err_entry["errors"])
                elif err_entry["record_id"] == "v-bad2":
                    assert any("source" in e.lower() for e in err_entry["errors"])
                elif err_entry["record_id"] == "v-bad3":
                    assert any("url" in e.lower() for e in err_entry["errors"])
        finally:
            conn.close()

    def test_field_presence_percentages(self) -> None:
        """Given a DB where optional fields are present in some but not all records,
        When verify_phase0_data is called,
        Then field_presence reports correct counts and percentages."""
        records = [
            _make_record_with_optional(
                "v-1", description="desc1", employer={"name": "A"}
            ),
            _make_record_with_optional(
                "v-2", description="desc2"
            ),
            _make_record_with_optional("v-3"),  # no optionals
            _make_record_with_optional(
                "v-4", description="desc4", employer={"name": "B"}, area={"name": "C"}
            ),
        ]
        conn = _make_in_memory_conn_with(records)
        try:
            report = verify_phase0_data(conn)
            presence = report["field_presence"]

            # description present in v-1, v-2, v-4 = 3/4
            assert presence["description"]["present"] == 3
            assert presence["description"]["absent"] == 1
            assert presence["description"]["percentage"] == pytest.approx(75.0)

            # employer present in v-1, v-4 = 2/4
            assert presence["employer"]["present"] == 2
            assert presence["employer"]["absent"] == 2
            assert presence["employer"]["percentage"] == pytest.approx(50.0)

            # area present in only v-4 = 1/4
            assert presence["area"]["present"] == 1
            assert presence["area"]["absent"] == 3
            assert presence["area"]["percentage"] == pytest.approx(25.0)

            # salary present in none = 0/4
            assert presence["salary"]["present"] == 0
            assert presence["salary"]["absent"] == 4
            assert presence["salary"]["percentage"] == pytest.approx(0.0)

            # All optional fields are represented
            for field in OPTIONAL_FIELDS:
                assert field in presence
        finally:
            conn.close()

    def test_date_validity_rejects_bad_timestamps(self) -> None:
        """Given a DB with invalid _phase0_capture_ts values,
        When verify_phase0_data is called,
        Then date_validity reports them."""
        records = [
            _make_record_with_optional("v-1", **{  # type: ignore[arg-type]
                "_phase0_capture_ts": "2024-01-15T12:00:00"  # valid
            }),
            {
                "id": "v-2",
                "name": "Bad date",
                "_phase0_source": "wayback-hhru",
                "_phase0_capture_ts": "not-a-date",
                "_phase0_original_url": "https://hh.ru/vacancy/2",
            },
            {
                "id": "v-3",
                "name": "Empty date",
                "_phase0_source": "wayback-linkedin",
                "_phase0_capture_ts": "",
                "_phase0_original_url": "https://linkedin.com/jobs/view/3",
            },
            {
                "id": "v-4",
                "name": "None date",
                "_phase0_source": "trudvsem",
                "_phase0_capture_ts": None,
                "_phase0_original_url": "https://trudvsem.ru/v/4",
            },
            {
                "id": "v-5",
                "name": "Missing date",
                "_phase0_source": "rostud",
                "_phase0_original_url": "https://rostud.gov.ru/5",
            },
        ]
        conn = _make_in_memory_conn_with(records)
        try:
            report = verify_phase0_data(conn)

            dv = report["date_validity"]
            assert dv["invalid_capture_ts"] == 4  # v-2, v-3, v-4, v-5

            invalid_ids = {d["record_id"] for d in dv["details"]}
            assert invalid_ids == {"v-2", "v-3", "v-4", "v-5"}
        finally:
            conn.close()

    def test_date_validity_accepts_various_iso_formats(self) -> None:
        """Given records with various ISO date formats,
        When verify_phase0_data is called,
        Then all accepted formats are valid."""
        records = [
            _make_record_with_optional("v-1", **{  # type: ignore[arg-type]
                "_phase0_capture_ts": "2024-01-15"  # date only
            }),
            _make_record_with_optional("v-2", **{  # type: ignore[arg-type]
                "_phase0_capture_ts": "2024-01-15T12:00:00"  # datetime
            }),
            _make_record_with_optional("v-3", **{  # type: ignore[arg-type]
                "_phase0_capture_ts": "20240101000000"  # compact format
            }),
        ]
        conn = _make_in_memory_conn_with(records)
        try:
            report = verify_phase0_data(conn)
            assert report["date_validity"]["invalid_capture_ts"] == 0
        finally:
            conn.close()

    def test_dedup_detects_duplicate_ids(self) -> None:
        """Given a DB with records sharing the same data-level id,
        When verify_phase0_data is called,
        Then dedup reports duplicate_ids and duplicate_count."""
        import json as _json

        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE raw_vacancies (
                id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                data JSON NOT NULL,
                fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        records = [
            _make_valid_record("v-1"),
            _make_valid_record("v-1"),  # duplicate
            _make_valid_record("v-2"),
            _make_valid_record("v-1"),  # triplicate
            _make_valid_record("v-3"),
            _make_valid_record("v-3"),  # duplicate
        ]
        for i, rec in enumerate(records):
            conn.execute(
                "INSERT INTO raw_vacancies (id, run_id, data) VALUES (?, ?, ?)",
                [f"row-{i}", "test-run", _json.dumps(rec, ensure_ascii=False)],
            )
        try:
            report = verify_phase0_data(conn)

            dedup = report["dedup"]
            assert set(dedup["duplicate_ids"]) == {"v-1", "v-3"}
            # v-1 appears 3 times → 2 extra, v-3 appears 2 times → 1 extra
            assert dedup["duplicate_count"] == 3
        finally:
            conn.close()

    def test_empty_db_produces_zero_report(self) -> None:
        """Given an empty raw_vacancies table,
        When verify_phase0_data is called,
        Then summary shows 0 records and field_presence has 0 percentages."""
        conn = duckdb.connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_vacancies (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                data JSON NOT NULL,
                fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            report = verify_phase0_data(conn)

            assert report["summary"]["total_records"] == 0
            assert report["summary"]["valid_records"] == 0
            assert report["summary"]["invalid_records"] == 0
            assert report["schema_compliance"]["error_count"] == 0
            assert report["date_validity"]["invalid_capture_ts"] == 0
            assert report["dedup"]["duplicate_count"] == 0

            for field in OPTIONAL_FIELDS:
                assert report["field_presence"][field]["percentage"] == 0.0
        finally:
            conn.close()


# --- Tests: save_verification_report -------------------------------------------


class TestSaveVerificationReport:
    def test_writes_valid_json(self) -> None:
        """Given a verification report dict,
        When save_verification_report is called,
        Then a valid JSON file is written."""
        report = {
            "summary": {"total_records": 1, "valid_records": 1, "invalid_records": 0},
            "schema_compliance": {"error_count": 0, "errors": []},
            "field_presence": {"description": {"present": 1, "absent": 0, "percentage": 100.0}},
            "date_validity": {"invalid_capture_ts": 0, "details": []},
            "dedup": {"duplicate_ids": [], "duplicate_count": 0},
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name

        try:
            save_verification_report(report, out_path)
            with open(out_path, encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded == report
            assert loaded["summary"]["total_records"] == 1
            assert loaded["schema_compliance"]["error_count"] == 0
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_creates_parent_directories(self) -> None:
        """Given a path with non-existent parent dirs,
        When save_verification_report is called,
        Then parent dirs are created and the file is written."""
        report = {"summary": {"total_records": 0}}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "nested" / "dir" / "report.json"
            save_verification_report(report, out_path)
            assert out_path.exists()
            with open(out_path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == report

    def test_unicode_in_report(self) -> None:
        """Given a report with Cyrillic content,
        When save_verification_report is called,
        Then the file contains properly encoded Cyrillic text."""
        report = {
            "summary": {"total_records": 1},
            "schema_compliance": {
                "error_count": 1,
                "errors": [{"record_id": "v-1", "errors": ["Отсутствует поле name"]}],
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name

        try:
            save_verification_report(report, out_path)
            with open(out_path, encoding="utf-8") as f:
                loaded = json.load(f)

            assert "Отсутствует поле name" in loaded["schema_compliance"]["errors"][0]["errors"]
        finally:
            Path(out_path).unlink(missing_ok=True)


# --- Tests: verify_phase0_db ---------------------------------------------------


class TestVerifyPhase0Db:
    def test_convenience_function_works(self) -> None:
        """Given a Phase 0 DuckDB file path,
        When verify_phase0_db is called,
        Then it opens the DB, runs verification, and returns a report."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)

            conn = duckdb.connect(db_path)
            _populate_db(conn, [
                _make_valid_record("v-1"),
                _make_valid_record("v-2"),
                _make_valid_record("v-3"),
            ])
            conn.close()
            conn = None

            report = verify_phase0_db(db_path)

            assert report["summary"]["total_records"] == 3
            assert report["summary"]["valid_records"] == 3
            assert report["summary"]["invalid_records"] == 0
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# --- Tests: field_presence_all_optionals ---------------------------------------


class TestAllOptionalsInReport:
    def test_all_optional_fields_appear_in_presence_report(self) -> None:
        """Given a DB with records,
        When verify_phase0_data is called,
        Then field_presence has an entry for every optional field from the schema."""
        conn = _make_in_memory_conn_with([_make_valid_record("v-1")])
        try:
            report = verify_phase0_data(conn)
            presence_keys = set(report["field_presence"].keys())

            # All optional fields must be represented
            for field in OPTIONAL_FIELDS:
                assert field in presence_keys, f"Missing field '{field}' in presence report"
        finally:
            conn.close()
