"""Tests for Phase 0 DuckDB storage layer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb
import pytest

from src.krm.phase0.storage import (
    finish_phase0_run,
    get_phase0_connection,
    init_phase0_tables,
    merge_into_main,
    start_phase0_run,
    upsert_phase0_vacancy,
)


# --- Helpers ------------------------------------------------------------------


def _table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return a set of column names for a table."""
    result = conn.execute(f"DESCRIBE {table}").fetchall()
    return {row[0] for row in result}


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Check if a table exists."""
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    return bool(result[0])


def _row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Return the number of rows in a table."""
    result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return result[0]


# --- Tests: Connection --------------------------------------------------------


class TestGetPhase0Connection:
    def test_creates_db_with_explicit_path(self) -> None:
        """Given an explicit path to a non-existent db,
        When get_phase0_connection is called,
        Then it creates the parent dirs and returns a usable connection."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = get_phase0_connection(db_path)
            assert conn is not None
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_reuses_existing_db(self) -> None:
        """Given an existing DuckDB file,
        When get_phase0_connection is called on it,
        Then it opens the existing database with its data preserved."""
        db_path = None
        conn1 = None
        conn2 = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)

            conn1 = get_phase0_connection(db_path)
            conn1.execute("CREATE TABLE test_table (val INTEGER)")
            conn1.execute("INSERT INTO test_table VALUES (42)")
            conn1.close()
            conn1 = None

            conn2 = get_phase0_connection(db_path)
            result = conn2.execute("SELECT val FROM test_table").fetchone()
            assert result[0] == 42
        finally:
            if conn1 is not None:
                conn1.close()
            if conn2 is not None:
                conn2.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# --- Tests: Table initialization ----------------------------------------------


class TestInitPhase0Tables:
    def test_creates_both_tables(self) -> None:
        """Given a fresh Phase 0 database,
        When init_phase0_tables is called,
        Then both raw_vacancies and phase0_runs tables exist."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            assert _table_exists(conn, "raw_vacancies")
            assert _table_exists(conn, "phase0_runs")
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_is_idempotent(self) -> None:
        """Given a database where tables already exist,
        When init_phase0_tables is called again,
        Then it does not raise an error."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)
            init_phase0_tables(conn)  # Should not raise
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_raw_vacancies_schema(self) -> None:
        """Given init_phase0_tables called,
        When checking raw_vacancies columns,
        Then it has id, run_id, data, fetched_at columns."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            columns = _table_columns(conn, "raw_vacancies")
            assert columns == {"id", "run_id", "data", "fetched_at"}
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_phase0_runs_schema(self) -> None:
        """Given init_phase0_tables called,
        When checking phase0_runs columns,
        Then it has the expected 8 columns."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            columns = _table_columns(conn, "phase0_runs")
            assert columns == {
                "run_id",
                "source",
                "source_url_pattern",
                "started_at",
                "completed_at",
                "records_fetched",
                "records_stored",
                "params",
            }
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# --- Tests: Upsert ------------------------------------------------------------


class TestUpsertPhase0Vacancy:
    def test_inserts_vacancy_with_json_data(self) -> None:
        """Given init_phase0_tables called,
        When upsert_phase0_vacancy inserts a record,
        Then the record is retrievable with correct JSON data."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            data = {"title": "физик-экспериментатор", "salary": 120000}
            upsert_phase0_vacancy(conn, "run-001", "vac-abc", data)

            row = conn.execute(
                "SELECT id, run_id, data, fetched_at FROM raw_vacancies WHERE id = 'vac-abc'"
            ).fetchone()
            assert row[0] == "vac-abc"
            assert row[1] == "run-001"
            assert json.loads(row[2]) == data
            assert row[3] is not None  # fetched_at is auto-populated
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_ignores_duplicate_inserts(self) -> None:
        """Given a vacancy already inserted,
        When upsert_phase0_vacancy is called again with the same id,
        Then the original record is preserved and no error is raised."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            data1 = {"title": "физик", "salary": 100000}
            data2 = {"title": "физик-v2", "salary": 150000}

            upsert_phase0_vacancy(conn, "run-001", "vac-dup", data1)
            upsert_phase0_vacancy(conn, "run-002", "vac-dup", data2)

            assert _row_count(conn, "raw_vacancies") == 1
            row = conn.execute(
                "SELECT data FROM raw_vacancies WHERE id = 'vac-dup'"
            ).fetchone()
            assert json.loads(row[0]) == data1  # First insert wins
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_handles_unicode_data(self) -> None:
        """Given vacancy data with Cyrillic and special chars,
        When upsert_phase0_vacancy stores it,
        Then the data is stored and retrieved correctly."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            data = {
                "title": "младший научный сотрудник",
                "description": "Эксперименты с низкими температурами (< 4K)",
                "emoji": "🔬",
            }
            upsert_phase0_vacancy(conn, "run-001", "vac-unicode", data)

            row = conn.execute(
                "SELECT data FROM raw_vacancies WHERE id = 'vac-unicode'"
            ).fetchone()
            retrieved = json.loads(row[0])
            assert retrieved == data
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# --- Tests: Run lifecycle -----------------------------------------------------


class TestRunLifecycle:
    def test_start_run_creates_record(self) -> None:
        """Given init_phase0_tables called,
        When start_phase0_run is called,
        Then a run record is inserted with correct fields."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            start_phase0_run(
                conn,
                run_id="run-2024-01",
                source="hhru_wayback",
                url_pattern="hh.ru/vacancy/*",
                params={"year": 2024, "region": 1},
            )

            row = conn.execute(
                "SELECT run_id, source, source_url_pattern, started_at, params "
                "FROM phase0_runs WHERE run_id = 'run-2024-01'"
            ).fetchone()

            assert row[0] == "run-2024-01"
            assert row[1] == "hhru_wayback"
            assert row[2] == "hh.ru/vacancy/*"
            assert row[3] is not None  # started_at
            assert json.loads(row[4]) == {"year": 2024, "region": 1}
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_start_run_without_optional_fields(self) -> None:
        """Given init_phase0_tables called,
        When start_phase0_run is called with only required args,
        Then optional fields are NULL."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            start_phase0_run(conn, run_id="run-minimal", source="trudvsem")

            row = conn.execute(
                "SELECT source_url_pattern, params FROM phase0_runs WHERE run_id = 'run-minimal'"
            ).fetchone()
            assert row[0] is None  # source_url_pattern
            assert row[1] is None  # params
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_finish_run_updates_fields(self) -> None:
        """Given a started run,
        When finish_phase0_run is called,
        Then completed_at and counters are updated."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            start_phase0_run(conn, "run-001", "linkedin_wayback")
            finish_phase0_run(conn, "run-001", records_fetched=500, records_stored=480)

            row = conn.execute(
                "SELECT records_fetched, records_stored, completed_at "
                "FROM phase0_runs WHERE run_id = 'run-001'"
            ).fetchone()
            assert row[0] == 500
            assert row[1] == 480
            assert row[2] is not None  # completed_at
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_full_run_workflow(self) -> None:
        """Given a complete run lifecycle: start → upsert → finish,
        When all steps complete,
        Then run metadata and vacancy data are consistent."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            run_id = "run-full-001"
            start_phase0_run(conn, run_id, "hhru_wayback")

            vacancies = [("v-001", {"title": "физик"}), ("v-002", {"title": "химик"})]
            for vid, data in vacancies:
                upsert_phase0_vacancy(conn, run_id, vid, data)

            finish_phase0_run(conn, run_id, records_fetched=2, records_stored=2)

            # Verify run
            run = conn.execute(
                "SELECT records_fetched, records_stored FROM phase0_runs WHERE run_id = ?",
                [run_id],
            ).fetchone()
            assert run[0] == 2
            assert run[1] == 2

            # Verify vacancies belong to the run
            count = conn.execute(
                "SELECT COUNT(*) FROM raw_vacancies WHERE run_id = ?", [run_id]
            ).fetchone()
            assert count[0] == 2
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# --- Tests: Merge into main ---------------------------------------------------


class TestMergeIntoMain:
    def test_merges_vacancies_into_empty_main_db(self) -> None:
        """Given a Phase 0 DB with vacancies and an empty main DB,
        When merge_into_main is called,
        Then all vacancies appear in the main DB."""
        phase0_path = None
        main_path = None
        try:
            # Create Phase 0 DB with vacancies
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                phase0_path = f.name
            Path(phase0_path).unlink(missing_ok=True)

            conn0 = duckdb.connect(phase0_path)
            init_phase0_tables(conn0)
            start_phase0_run(conn0, "run-001", "hhru_wayback")
            upsert_phase0_vacancy(conn0, "run-001", "v-1", {"title": "физик"})
            upsert_phase0_vacancy(conn0, "run-001", "v-2", {"title": "химик"})
            conn0.close()

            # Create empty main DB
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                main_path = f.name
            Path(main_path).unlink(missing_ok=True)

            merged = merge_into_main(phase0_path, main_path)
            assert merged == 2

            # Verify main DB has the data
            conn_main = duckdb.connect(main_path)
            assert _table_exists(conn_main, "raw_vacancies")
            assert _row_count(conn_main, "raw_vacancies") == 2

            rows = conn_main.execute(
                "SELECT id, run_id FROM raw_vacancies ORDER BY id"
            ).fetchall()
            assert rows[0] == ("v-1", "run-001")
            assert rows[1] == ("v-2", "run-001")
            conn_main.close()
        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)

    def test_merge_is_idempotent(self) -> None:
        """Given Phase 0 data already merged into main DB,
        When merge_into_main is called again,
        Then no duplicates are created (INSERT OR IGNORE)."""
        phase0_path = None
        main_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                phase0_path = f.name
            Path(phase0_path).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                main_path = f.name
            Path(main_path).unlink(missing_ok=True)

            conn0 = duckdb.connect(phase0_path)
            init_phase0_tables(conn0)
            start_phase0_run(conn0, "run-001", "hhru_wayback")
            upsert_phase0_vacancy(conn0, "run-001", "v-only", {"title": "единственный"})
            conn0.close()

            # First merge
            first = merge_into_main(phase0_path, main_path)
            assert first == 1

            # Second merge - should ignore duplicate
            second = merge_into_main(phase0_path, main_path)
            assert second == 0  # No new rows inserted

            conn_main = duckdb.connect(main_path)
            assert _row_count(conn_main, "raw_vacancies") == 1
            conn_main.close()
        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)

    def test_empty_merge_returns_zero(self) -> None:
        """Given a Phase 0 DB with no vacancies,
        When merge_into_main is called,
        Then 0 is returned and main DB table exists but is empty."""
        phase0_path = None
        main_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                phase0_path = f.name
            Path(phase0_path).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                main_path = f.name
            Path(main_path).unlink(missing_ok=True)

            conn0 = duckdb.connect(phase0_path)
            init_phase0_tables(conn0)
            conn0.close()

            merged = merge_into_main(phase0_path, main_path)
            assert merged == 0

            conn_main = duckdb.connect(main_path)
            assert _table_exists(conn_main, "raw_vacancies")
            assert _row_count(conn_main, "raw_vacancies") == 0
            conn_main.close()
        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)

    def test_does_not_overwrite_existing_main_vacancies(self) -> None:
        """Given main DB already has a vacancy with id 'v-1',
        When Phase 0 data with same id is merged,
        Then the existing main DB record is preserved."""
        phase0_path = None
        main_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                phase0_path = f.name
            Path(phase0_path).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                main_path = f.name
            Path(main_path).unlink(missing_ok=True)

            # Phase 0 DB with v-1 data
            conn0 = duckdb.connect(phase0_path)
            init_phase0_tables(conn0)
            start_phase0_run(conn0, "run-p0", "hhru_wayback")
            upsert_phase0_vacancy(conn0, "run-p0", "v-1", {"title": "from-phase0"})
            conn0.close()

            # Main DB with pre-existing v-1
            conn_m = duckdb.connect(main_path)
            conn_m.execute("""
                CREATE TABLE raw_vacancies (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    data JSON NOT NULL,
                    fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn_m.execute(
                "INSERT INTO raw_vacancies (id, run_id, data) VALUES (?, ?, ?)",
                ["v-1", "run-main", json.dumps({"title": "from-main"})],
            )
            conn_m.close()

            merged = merge_into_main(phase0_path, main_path)
            assert merged == 0  # Nothing inserted (all ignored)

            conn_main = duckdb.connect(main_path)
            row = conn_main.execute(
                "SELECT run_id, data FROM raw_vacancies WHERE id = 'v-1'"
            ).fetchone()
            assert row[0] == "run-main"  # Original preserved
            assert json.loads(row[1]) == {"title": "from-main"}
            conn_main.close()
        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)
