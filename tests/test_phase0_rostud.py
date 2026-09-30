"""Tests for Rostrud bulk CSV collector (phase0/collectors/rostud.py)."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from krm.phase0.collectors.rostud import collect_rostud


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockConfig:
    """Minimal duck-type config for collect_rostud()."""

    phase0_rostud_enabled: bool = True
    phase0_rostud_dataset_path: str = ""
    phase0_db_path: str = ""


def _make_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as a CSV file."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_many_rows(path: Path, n: int) -> None:
    """Write `n` rows with Russian column names to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "Должность", "Описание", "Работодатель", "Регион",
        ])
        for i in range(1, n + 1):
            writer.writerow([
                str(i),
                f"Должность {i}",
                f"Описание {i}",
                f"Работодатель {i}",
                f"Регион {i}",
            ])


def _row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Return the number of rows in a table."""
    result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert result is not None
    return result[0]


# ---------------------------------------------------------------------------
# Russian column names
# ---------------------------------------------------------------------------


class TestRussianColumnNames:
    """Rows with Russian column names (Должность, Описание, etc.)."""

    def test_russian_columns_stored(self) -> None:
        """Given a CSV with Russian header names,
        When collect_rostud runs,
        Then rows are mapped and stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {
                    "id": "1",
                    "Должность": "Инженер",
                    "Описание": "Разработка",
                    "Работодатель": "Росатом",
                    "Регион": "Москва",
                },
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1

            # Verify storage
            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            assert len(stored) == 1
            data = json.loads(stored[0][0])
            assert data["name"] == "Инженер"
            assert data["description"] == "Разработка"
            assert data["employer"] == {"name": "Росатом"}
            assert data["area"] == {"name": "Москва"}
            assert data["_phase0_source"] == "rostud"

    def test_multiple_russian_rows(self) -> None:
        """Given a CSV with multiple Russian-header rows,
        When collect_rostud runs,
        Then all rows are stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {"id": "1", "Должность": "Химик", "Описание": "", "Работодатель": "", "Регион": ""},
                {"id": "2", "Должность": "Биолог", "Описание": "", "Работодатель": "", "Регион": ""},
                {"id": "3", "Должность": "Физик", "Описание": "", "Работодатель": "", "Регион": ""},
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 3


# ---------------------------------------------------------------------------
# English column names
# ---------------------------------------------------------------------------


class TestEnglishColumnNames:
    """Rows with English/latin column names (name, description, etc.)."""

    def test_english_name_description_company_location(self) -> None:
        """Given a CSV with English headers (name, description, company, location),
        When collect_rostud runs,
        Then the mapper uses latin field fallbacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {
                    "id": "1",
                    "name": "Scientist",
                    "description": "Research work",
                    "company": "CERN",
                    "location": "Geneva",
                },
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1

            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            data = json.loads(stored[0][0])
            assert data["name"] == "Scientist"
            assert data["description"] == "Research work"
            assert data["employer"] == {"name": "CERN"}
            assert data["area"] == {"name": "Geneva"}

    def test_english_position_employer_region(self) -> None:
        """Given a CSV with position/employer/region (alternate English names),
        When collect_rostud runs,
        Then fallback column names are used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {
                    "id": "1",
                    "position": "Lab Technician",
                    "employer": "MIT",
                    "region": "Boston",
                },
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1

            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            data = json.loads(stored[0][0])
            assert data["name"] == "Lab Technician"
            assert data["employer"] == {"name": "MIT"}
            assert data["area"] == {"name": "Boston"}


# ---------------------------------------------------------------------------
# Mixed / missing columns
# ---------------------------------------------------------------------------


class TestMixedAndMissingColumns:
    """Rows with a mix of Russian/English names, or completely missing fields."""

    def test_mixed_russian_and_english_columns(self) -> None:
        """Given a CSV with some Russian columns and some English columns,
        When collect_rostud runs,
        Then the mapper picks whichever is present, Russian preferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {
                    "id": "1",
                    "Должность": "Физик",
                    "description": "English fallback desc",
                    "company": "English company name",
                    "Регион": "Новосибирск",
                },
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1

            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            data = json.loads(stored[0][0])
            assert data["name"] == "Физик"
            assert data["description"] == "English fallback desc"
            assert data["employer"] == {"name": "English company name"}
            assert data["area"] == {"name": "Новосибирск"}

    def test_row_with_only_id(self) -> None:
        """Given a CSV row with only an 'id' field and no name,
        When collect_rostud runs,
        Then the row is skipped (name is empty, normalize_record raises ValueError)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {"id": "1", "name": ""},
                {"id": "2", "name": "Valid Job"},
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1

            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            assert len(stored) == 1
            data = json.loads(stored[0][0])
            assert data["name"] == "Valid Job"

    def test_missing_description_column(self) -> None:
        """Given rows without a description column,
        When collect_rostud runs,
        Then description is set to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {
                    "id": "1",
                    "Должность": "Лаборант",
                    "Работодатель": "МГУ",
                    "Регион": "Москва",
                },
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1

            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            data = json.loads(stored[0][0])
            assert data["name"] == "Лаборант"
            assert data["description"] is None

    def test_id_hash_fallback_when_id_missing(self) -> None:
        """Given rows without an 'id' column but with a name,
        When collect_rostud runs,
        Then a stable hash-based ID is generated by map_rostud."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {"Должность": "Химик", "Работодатель": "НИИ", "Регион": "Казань"},
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1

            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT id, data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            vid = stored[0][0]
            # ID should be a non-empty string (hash-generated)
            assert vid
            assert isinstance(vid, str)

    def test_url_and_source_url_handled(self) -> None:
        """Given rows with url or source_url columns,
        When collect_rostud runs,
        Then _phase0_original_url is populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {"id": "1", "name": "Job", "url": "https://example.com/v/1"},
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            collect_rostud(cfg)

            conn = duckdb.connect(db_path)
            stored = conn.execute(
                "SELECT data FROM raw_vacancies"
            ).fetchall()
            conn.close()
            data = json.loads(stored[0][0])
            assert data["_phase0_original_url"] == "https://example.com/v/1"


# ---------------------------------------------------------------------------
# Chunked reading
# ---------------------------------------------------------------------------


class TestChunkedReading:
    """CSV chunked reading with chunksize=10000."""

    def test_chunked_reading_multiple_chunks(self) -> None:
        """Given a CSV with 15 000 rows,
        When collect_rostud runs with chunksize=10 000,
        Then two chunks are processed and all 15 000 rows stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _write_many_rows(csv_path, 15_000)

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 15_000

            # Verify the count in DuckDB matches
            conn = duckdb.connect(db_path)
            count = _row_count(conn, "raw_vacancies")
            conn.close()
            assert count == 15_000

    def test_chunked_reading_exact_chunk_boundary(self) -> None:
        """Given a CSV with exactly 10 000 rows (the chunk size),
        When collect_rostud runs,
        Then one chunk is processed and all rows stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _write_many_rows(csv_path, 10_000)

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 10_000

    def test_chunked_reading_single_row(self) -> None:
        """Given a CSV with 1 row,
        When collect_rostud runs,
        Then it is stored correctly (chunked reader handles small files)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {"id": "1", "name": "Only Job"},
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 1


# ---------------------------------------------------------------------------
# Disabled collector
# ---------------------------------------------------------------------------


class TestDisabledCollector:
    """When config.phase0_rostud_enabled is False."""

    def test_disabled_returns_zero(self) -> None:
        """Given enabled=False,
        When collect_rostud is called,
        Then it prints a message and returns 0 without accessing storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            cfg = _MockConfig(
                phase0_rostud_enabled=False,
                phase0_rostud_dataset_path="/nonexistent",
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 0

            # Verify no database was created (or it's empty)
            db_file = Path(db_path)
            if db_file.exists():
                conn = duckdb.connect(db_path)
                tables = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables"
                ).fetchone()
                conn.close()
                assert tables is not None and tables[0] == 0


# ---------------------------------------------------------------------------
# No files found
# ---------------------------------------------------------------------------


class TestNoFilesFound:
    """When the dataset directory has no CSV/XLSX files."""

    def test_empty_directory_returns_zero(self) -> None:
        """Given a dataset directory with no CSV/XLSX files,
        When collect_rostud runs,
        Then it returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            # Create a non-CSV file to verify it's ignored
            (dataset_dir / "README.txt").write_text("not a CSV")

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 0

    def test_non_csv_files_ignored(self) -> None:
        """Given a directory with .txt and .json files but no CSV/XLSX,
        When collect_rostud runs,
        Then none are processed and 0 is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            (dataset_dir / "data.txt").write_text("not CSV")
            (dataset_dir / "data.json").write_text('{"key": "value"}')

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 0

    def test_nonexistent_directory_returns_zero(self) -> None:
        """Given a dataset path that doesn't exist,
        When collect_rostud runs,
        Then it returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            cfg = _MockConfig(
                phase0_rostud_dataset_path="/nonexistent/path/12345",
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 0


# ---------------------------------------------------------------------------
# Storage integration
# ---------------------------------------------------------------------------


class TestStorageIntegration:
    """Integration with DuckDB storage (phase0_runs + raw_vacancies tables)."""

    def test_run_id_registered_in_phase0_runs(self) -> None:
        """Given a CSV file named 'data.csv',
        When collect_rostud runs,
        Then phase0_runs contains an entry with run_id='rostud-data'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "data.csv"
            _make_csv(csv_path, [
                {"id": "1", "name": "Job"},
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            collect_rostud(cfg)

            conn = duckdb.connect(db_path)
            runs = conn.execute(
                "SELECT run_id, source, records_fetched, records_stored"
                + " FROM phase0_runs"
            ).fetchall()
            conn.close()

            assert len(runs) == 1
            assert runs[0][0] == "rostud-data"
            assert runs[0][1] == "rostud"
            assert runs[0][2] == 1  # records_fetched
            assert runs[0][3] == 1  # records_stored

    def test_multiple_files_multiple_runs(self) -> None:
        """Given two CSV files,
        When collect_rostud runs,
        Then two runs are recorded with correct counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            _make_csv(dataset_dir / "a.csv", [
                {"id": "1", "name": "A1"},
                {"id": "2", "name": "A2"},
            ])
            _make_csv(dataset_dir / "b.csv", [
                {"id": "10", "name": "B1"},
                {"id": "11", "name": "B2"},
                {"id": "12", "name": "B3"},
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 5

            conn = duckdb.connect(db_path)
            runs = conn.execute(
                "SELECT run_id, records_fetched, records_stored"
                + " FROM phase0_runs ORDER BY run_id"
            ).fetchall()
            raw_count = _row_count(conn, "raw_vacancies")
            conn.close()

            assert len(runs) == 2
            assert runs[0][0] == "rostud-a"
            assert runs[0][1] == 2
            assert runs[0][2] == 2
            assert runs[1][0] == "rostud-b"
            assert runs[1][1] == 3
            assert runs[1][2] == 3
            assert raw_count == 5

    def test_raw_vacancy_data_format(self) -> None:
        """Given a stored vacancy,
        When queried from raw_vacancies,
        Then the data JSON contains all expected hhru-schema fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            _make_csv(dataset_dir / "data.csv", [
                {
                    "id": "v1",
                    "Должность": "Научный сотрудник",
                    "Описание": "Исследования",
                    "Работодатель": "ИПХФ РАН",
                    "Регион": "Черноголовка",
                },
            ])

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            collect_rostud(cfg)

            conn = duckdb.connect(db_path)
            rows = conn.execute(
                "SELECT id, run_id, data FROM raw_vacancies"
            ).fetchall()
            conn.close()

            assert len(rows) == 1
            assert rows[0][0] == "v1"
            assert rows[0][1] == "rostud-data"
            data = json.loads(rows[0][2])
            assert data["id"] == "v1"
            assert data["name"] == "Научный сотрудник"
            assert data["description"] == "Исследования"
            assert data["employer"] == {"name": "ИПХФ РАН"}
            assert data["area"] == {"name": "Черноголовка"}
            assert data["_phase0_source"] == "rostud"
            assert data["_phase0_capture_ts"] == "rostud"

    def test_empty_csv_file_handled(self) -> None:
        """Given an empty CSV file (header only),
        When collect_rostud runs,
        Then 0 records stored and run recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.duckdb")
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir()

            csv_path = dataset_dir / "empty.csv"
            csv_path.write_text("id,name\n", encoding="utf-8")

            cfg = _MockConfig(
                phase0_rostud_dataset_path=str(dataset_dir),
                phase0_db_path=db_path,
            )
            result = collect_rostud(cfg)
            assert result == 0
