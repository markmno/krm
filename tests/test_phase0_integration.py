"""Cross-phase integration tests using Phase 0 stored data.

Simulates a real collection scenario: multiple Phase 0 runs from different
sources produce raw vacancies, which are then merged into the main pipeline
database for downstream phase consumption.

Tests:
* Multi-source, multi-run collection flow
* Data queryability and cross-source consistency
* merge_into_main() as the Phase 0 → Phase 1 bridge
* JSON data integrity and schema compliance
* Run lifecycle tracking across concurrent sources
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb
import pytest

from krm.phase0.schema import (
    PHASE0_META_FIELDS,
    SOURCE_TYPES,
    normalize_record,
    validate_record,
)
from krm.phase0.storage import (
    finish_phase0_run,
    init_phase0_tables,
    merge_into_main,
    start_phase0_run,
    upsert_phase0_vacancy,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_vacancy_data(
    source: str,
    vacancy_id: str,
    name: str,
    capture_ts: str = "2024-01-01T00:00:00",
    original_url: str = "",
    **extra: object,
) -> dict[str, object]:
    """Build a valid normalized vacancy dict for storage."""
    data: dict[str, object] = {
        "id": vacancy_id,
        "name": name,
        "_phase0_source": source,
        "_phase0_capture_ts": capture_ts,
        "_phase0_original_url": original_url or f"https://example.com/{vacancy_id}",
        **extra,
    }
    return normalize_record(data)


def _vacancy_count(conn: duckdb.DuckDBPyConnection, table: str = "raw_vacancies") -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _distinct_sources(conn: duckdb.DuckDBPyConnection, table: str = "raw_vacancies") -> set[str]:
    rows = conn.execute(f"SELECT DISTINCT data FROM {table}").fetchall()
    return {json.loads(row[0]).get("_phase0_source", "") for row in rows}


# ---------------------------------------------------------------------------
# Multi-source, multi-run collection flow
# ---------------------------------------------------------------------------


class TestMultiRunCollection:
    """Simulate real collection: multiple runs from different sources."""

    def test_four_sources_two_runs_each(self) -> None:
        """When: 4 sources × 2 runs = 8 runs, each storing 1 vacancy,
        Then: 8 runs and 8 vacancies are recorded."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            for source in ["hhru_legacy", "hhru_modern", "linkedin_legacy", "linkedin_modern"]:
                for run_idx in range(2):
                    run_id = f"run-{source}-{run_idx:02d}"
                    vac = _make_vacancy_data(source, f"{source}-{run_idx:02d}", f"Job {source} #{run_idx}")
                    start_phase0_run(conn, run_id, source)
                    upsert_phase0_vacancy(conn, run_id, vac["id"], vac)
                    finish_phase0_run(conn, run_id, records_fetched=1, records_stored=1)

            assert _vacancy_count(conn) == 8
            run_rows = conn.execute("SELECT COUNT(*) FROM phase0_runs").fetchone()[0]
            assert run_rows == 8

            # All runs have completed_at set
            incomplete = conn.execute(
                "SELECT COUNT(*) FROM phase0_runs WHERE completed_at IS NULL"
            ).fetchone()[0]
            assert incomplete == 0, "All runs should be completed"

        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_different_vacancy_counts_per_run(self) -> None:
        """When: different runs store different numbers of vacancies,
        Then: per-run counts are accurately tracked."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            run_specs: list[tuple[str, str, int]] = [
                ("run-small", "hhru_modern", 1),
                ("run-medium", "hhru_modern", 3),
                ("run-large", "linkedin_modern", 5),
            ]

            for run_id, source, count in run_specs:
                start_phase0_run(conn, run_id, source)
                for i in range(count):
                    vac = _make_vacancy_data(source, f"{run_id}-{i:03d}", f"Job {i}")
                    upsert_phase0_vacancy(conn, run_id, vac["id"], vac)
                finish_phase0_run(conn, run_id, records_fetched=count, records_stored=count)

            # Verify per-run counts
            for run_id, _source, expected_count in run_specs:
                row = conn.execute(
                    "SELECT records_fetched, records_stored FROM phase0_runs WHERE run_id = ?",
                    [run_id],
                ).fetchone()
                assert row[0] == expected_count
                assert row[1] == expected_count

            assert _vacancy_count(conn) == 9  # 1 + 3 + 5

        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Cross-source data consistency
# ---------------------------------------------------------------------------


class TestCrossSourceConsistency:
    """Data from all 4 sources is queryable and schema-compliant."""

    @pytest.fixture
    def populated_db(self) -> tuple[str, duckdb.DuckDBPyConnection]:
        """Populated Phase 0 DB with one vacancy per source."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            sources: list[tuple[str, str, str | None, str | None]] = [
                ("wayback-hhru", "run-hhru", "Инженер-программист", "https://hh.ru/vacancy/111"),
                ("wayback-linkedin", "run-linkedin", "Software Engineer", "https://linkedin.com/jobs/222"),
                ("trudvsem", "run-trudvsem", "Программист", "https://trudvsem.ru/v/333"),
                ("rostud", "run-rostud", "Лаборант", "https://rostrud.gov.ru/r/444"),
            ]
            for source, run_id, name, url in sources:
                vac = _make_vacancy_data(source, f"id-{source}", name, original_url=url or "")
                start_phase0_run(conn, run_id, source)
                upsert_phase0_vacancy(conn, run_id, vac["id"], vac)
                finish_phase0_run(conn, run_id, records_fetched=1, records_stored=1)

            yield db_path, conn
        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_all_sources_queryable_by_source(self, populated_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: data from all 4 sources is stored, Then: each source can be filtered."""
        _db_path, conn = populated_db

        base_sources = {"wayback-hhru", "wayback-linkedin", "trudvsem", "rostud"}
        for source in base_sources:
            rows = conn.execute(
                "SELECT data FROM raw_vacancies WHERE json_extract_string(data, '$._phase0_source') = ?",
                [source],
            ).fetchall()
            assert len(rows) == 1, f"Expected 1 record for source {source}, got {len(rows)}"
            record = json.loads(rows[0][0])
            assert record["_phase0_source"] == source
            assert record["name"], f"{source}: name should not be empty"
            assert record["id"], f"{source}: id should not be empty"

    def test_all_vacancies_have_complete_meta(self, populated_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: all records are queried, Then: every record has all PHASE0_META_FIELDS."""
        _db_path, conn = populated_db

        rows = conn.execute("SELECT data FROM raw_vacancies").fetchall()
        assert len(rows) == 4

        for row in rows:
            record = json.loads(row[0])
            for field in PHASE0_META_FIELDS:
                assert field in record, f"Missing meta field {field} in record {record.get('id')}"

    def test_all_records_validate(self, populated_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: all stored records are validated, Then: no validation errors."""
        _db_path, conn = populated_db

        rows = conn.execute("SELECT data FROM raw_vacancies").fetchall()
        for row in rows:
            record = json.loads(row[0])
            errors = validate_record(record)
            assert errors == [], f"Validation errors for {record.get('id')}: {errors}"

    def test_cross_source_name_diversity(self, populated_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: all records are queried, Then: names are diverse (not all same)."""
        _db_path, conn = populated_db

        rows = conn.execute("SELECT data FROM raw_vacancies").fetchall()
        names = {json.loads(row[0]).get("name") for row in rows}
        assert len(names) == 4, f"Expected 4 distinct names, got {len(names)}: {names}"


# ---------------------------------------------------------------------------
# merge_into_main: Phase 0 → main pipeline bridge
# ---------------------------------------------------------------------------


class TestMergeIntoMain:
    """Cross-phase bridge: Phase 0 data merged into the main pipeline DB."""

    def test_merge_preserves_run_lineage(self) -> None:
        """When: Phase 0 data is merged into main, Then: run_id is preserved."""
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

            start_phase0_run(conn0, "run-alpha", "wayback-hhru")
            vac = _make_vacancy_data("wayback-hhru", "v-001", "Физик")
            upsert_phase0_vacancy(conn0, "run-alpha", "v-001", vac)
            finish_phase0_run(conn0, "run-alpha", records_fetched=1, records_stored=1)

            start_phase0_run(conn0, "run-beta", "wayback-linkedin")
            vac2 = _make_vacancy_data("wayback-linkedin", "v-002", "Engineer")
            upsert_phase0_vacancy(conn0, "run-beta", "v-002", vac2)
            finish_phase0_run(conn0, "run-beta", records_fetched=1, records_stored=1)
            conn0.close()

            merged = merge_into_main(phase0_path, main_path)
            assert merged == 2

            conn_main = duckdb.connect(main_path)
            rows = conn_main.execute(
                "SELECT id, run_id FROM raw_vacancies ORDER BY id"
            ).fetchall()
            assert rows[0] == ("v-001", "run-alpha")
            assert rows[1] == ("v-002", "run-beta")
            conn_main.close()

        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)

    def test_merge_handles_all_source_types(self) -> None:
        """When: all 4 source types are merged, Then: all are present in main DB."""
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

            for i, source in enumerate(("wayback-hhru", "wayback-linkedin", "trudvsem", "rostud")):
                run_id = f"run-{source}"
                start_phase0_run(conn0, run_id, source)
                vac = _make_vacancy_data(source, f"v-src-{i:03d}", f"Job from {source}")
                upsert_phase0_vacancy(conn0, run_id, f"v-src-{i:03d}", vac)
                finish_phase0_run(conn0, run_id, records_fetched=1, records_stored=1)
            conn0.close()

            merged = merge_into_main(phase0_path, main_path)
            assert merged == 4

            conn_main = duckdb.connect(main_path)
            stored_sources = _distinct_sources(conn_main)
            assert             stored_sources == {"wayback-hhru", "wayback-linkedin", "trudvsem", "rostud"}, f"Missing sources"
            conn_main.close()

        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)

    def test_merge_is_idempotent_across_runs(self) -> None:
        """When: same Phase 0 data is merged twice, Then: main DB count is stable."""
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
            start_phase0_run(conn0, "run-1", "wayback-hhru")
            for i in range(5):
                vac = _make_vacancy_data("wayback-hhru", f"v-{i:03d}", f"Job {i}")
                upsert_phase0_vacancy(conn0, "run-1", f"v-{i:03d}", vac)
            finish_phase0_run(conn0, "run-1", records_fetched=5, records_stored=5)
            conn0.close()

            first_merge = merge_into_main(phase0_path, main_path)
            assert first_merge == 5

            second_merge = merge_into_main(phase0_path, main_path)
            assert second_merge == 0  # No new rows

            conn_main = duckdb.connect(main_path)
            assert _vacancy_count(conn_main) == 5
            conn_main.close()

        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)

    def test_main_db_preserves_existing_records_on_remerge(self) -> None:
        """When: main DB has pre-existing records from another source,
        Then: re-merge does not overwrite them."""
        phase0_path = None
        main_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                phase0_path = f.name
            Path(phase0_path).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                main_path = f.name
            Path(main_path).unlink(missing_ok=True)

            # Phase 0 data
            conn0 = duckdb.connect(phase0_path)
            init_phase0_tables(conn0)
            start_phase0_run(conn0, "p0-run", "wayback-hhru")
            vac = _make_vacancy_data("wayback-hhru", "v-shared", "Hhru Job")
            upsert_phase0_vacancy(conn0, "p0-run", "v-shared", vac)
            finish_phase0_run(conn0, "p0-run", records_fetched=1, records_stored=1)
            conn0.close()

            # Pre-populate main DB with a record having the same id
            conn_main = duckdb.connect(main_path)
            conn_main.execute("""
                CREATE TABLE raw_vacancies (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    data JSON NOT NULL,
                    fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn_main.execute(
                "INSERT INTO raw_vacancies (id, run_id, data) VALUES (?, ?, ?)",
                ["v-shared", "original-run", json.dumps({"name": "Original", "id": "v-shared"})],
            )
            conn_main.close()

            # Merge — should ignore the conflicting id
            merged = merge_into_main(phase0_path, main_path)
            assert merged == 0

            # Verify original record preserved
            conn_main = duckdb.connect(main_path)
            row = conn_main.execute(
                "SELECT run_id, data FROM raw_vacancies WHERE id = 'v-shared'"
            ).fetchone()
            assert row[0] == "original-run"
            assert json.loads(row[1])["name"] == "Original"
            conn_main.close()

        finally:
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Data integrity: Round-trip JSON fidelity
# ---------------------------------------------------------------------------


class TestJsonFidelity:
    """Stored JSON data survives round-trips with full fidelity."""

    def test_cyrillic_and_unicode_roundtrip(self) -> None:
        """When: data with Cyrillic and emoji is stored and retrieved, Then: it's bit-exact."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            data = _make_vacancy_data(
                "wayback-hhru",
                "v-cyrillic",
                "младший научный сотрудник",
                original_url="https://hh.ru/vacancy/12345",
            )
            extra = {
                "description": "Эксперименты с низкими температурами (< 4K) и сверхпроводниками",
                "employer": {"name": "НИИ физики твёрдого тела"},
                "area": {"name": "Новосибирск"},
                "key_skills": [
                    {"name": "Python"},
                    {"name": "LabVIEW"},
                    {"name": "криогеника"},
                ],
            }
            data.update(extra)  # type: ignore[arg-type]

            start_phase0_run(conn, "run-utf8", "wayback-hhru")
            upsert_phase0_vacancy(conn, "run-utf8", "v-cyrillic", data)
            finish_phase0_run(conn, "run-utf8", records_fetched=1, records_stored=1)

            row = conn.execute(
                "SELECT data FROM raw_vacancies WHERE id = 'v-cyrillic'"
            ).fetchone()
            retrieved = json.loads(row[0])

            assert retrieved["name"] == "младший научный сотрудник"
            assert retrieved["description"] == extra["description"]
            assert retrieved["employer"]["name"] == "НИИ физики твёрдого тела"
            assert retrieved["area"]["name"] == "Новосибирск"
            assert retrieved["key_skills"] == extra["key_skills"]

        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_null_fields_preserved(self) -> None:
        """When: optional fields are None, Then: they remain None (not 'null' string)."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            data = normalize_record({
                "id": "v-minimal",
                "name": "Minimal Job",
                "_phase0_source": "rostud",
                "_phase0_capture_ts": "2024-01-01",
                "_phase0_original_url": "https://example.com/v-minimal",
            })

            start_phase0_run(conn, "run-null", "rostud")
            upsert_phase0_vacancy(conn, "run-null", "v-minimal", data)
            finish_phase0_run(conn, "run-null", records_fetched=1, records_stored=1)

            row = conn.execute(
                "SELECT data FROM raw_vacancies WHERE id = 'v-minimal'"
            ).fetchone()
            retrieved = json.loads(row[0])

            assert retrieved.get("description") is None
            assert retrieved.get("salary") is None
            assert retrieved.get("employer") is None
            assert retrieved.get("area") is None
            assert retrieved.get("experience") is None

        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Query patterns used by downstream phases
# ---------------------------------------------------------------------------


class TestDownstreamQueryPatterns:
    """SQL query patterns that Phase 1+ would use on Phase 0 data."""

    @pytest.fixture
    def populated_main_db(self) -> tuple[str, duckdb.DuckDBPyConnection]:
        """A merged main DB with 10 vacancies from all 4 sources."""
        phase0_path = None
        main_path = None
        conn_main = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                phase0_path = f.name
            Path(phase0_path).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                main_path = f.name
            Path(main_path).unlink(missing_ok=True)

            conn0 = duckdb.connect(phase0_path)
            init_phase0_tables(conn0)

            specs = [
                ("wayback-hhru", 3),
                ("wayback-linkedin", 3),
                ("trudvsem", 2),
                ("rostud", 2),
            ]
            idx = 0
            for source, count in specs:
                run_id = f"run-{source}"
                start_phase0_run(conn0, run_id, source)
                for _ in range(count):
                    vac = _make_vacancy_data(source, f"v-{idx:03d}", f"Job {idx} from {source}")
                    upsert_phase0_vacancy(conn0, run_id, f"v-{idx:03d}", vac)
                    idx += 1
                finish_phase0_run(conn0, run_id, records_fetched=count, records_stored=count)
            conn0.close()

            merge_into_main(phase0_path, main_path)

            conn_main = duckdb.connect(main_path)
            yield main_path, conn_main
        finally:
            if conn_main is not None:
                conn_main.close()
            for p in [phase0_path, main_path]:
                if p is not None:
                    Path(p).unlink(missing_ok=True)

    def test_count_by_source_group_by(self, populated_main_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: COUNT(*) GROUP BY _phase0_source, Then: correct per-source counts."""
        _db_path, conn = populated_main_db

        rows = conn.execute("""
            SELECT json_extract_string(data, '$._phase0_source') AS source, COUNT(*)
            FROM raw_vacancies
            GROUP BY source
            ORDER BY source
        """).fetchall()

        counts = {row[0]: row[1] for row in rows}
        assert counts.get("wayback-hhru") == 3
        assert counts.get("wayback-linkedin") == 3
        assert counts.get("trudvsem") == 2
        assert counts.get("rostud") == 2

    def test_filter_by_source(self, populated_main_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: filtering by _phase0_source, Then: only matching records returned."""
        _db_path, conn = populated_main_db

        for source in ("wayback-hhru", "wayback-linkedin", "trudvsem", "rostud"):
            rows = conn.execute(
                "SELECT data FROM raw_vacancies WHERE json_extract_string(data, '$._phase0_source') = ?",
                [source],
            ).fetchall()
            assert len(rows) > 0, f"No records for source {source}"
            for row in rows:
                assert json.loads(row[0])["_phase0_source"] == source

    def test_extract_names_as_json(self, populated_main_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: extracting name field from JSON data, Then: all names are present."""
        _db_path, conn = populated_main_db

        rows = conn.execute(
            "SELECT json_extract_string(data, '$.name') AS name FROM raw_vacancies ORDER BY id"
        ).fetchall()

        assert len(rows) == 10
        names = [row[0] for row in rows]
        assert all(isinstance(n, str) and len(n) > 0 for n in names), f"Empty/bad names: {names}"
        assert len(set(names)) == 10, "Expected 10 distinct names"

    def test_date_range_query_on_capture_ts(self, populated_main_db: tuple[str, duckdb.DuckDBPyConnection]) -> None:
        """When: querying by capture timestamp, Then: correct records returned."""
        _db_path, conn = populated_main_db

        all_rows = conn.execute(
            "SELECT json_extract_string(data, '$._phase0_capture_ts') FROM raw_vacancies"
        ).fetchall()
        assert len(all_rows) == 10
        for row in all_rows:
            assert row[0] == "2024-01-01T00:00:00", f"Unexpected capture_ts: {row[0]}"
