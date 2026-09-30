"""End-to-end pipeline tests for Phase 0: mock CDX → real parser → real mapper → real storage.

Tests the full pipeline for all 4 source types using HTML fixtures that
exercise the real parsers. No real CDX/API calls — CDX records are mocked
and HTML is read from ``tests/fixtures/phase0/``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb
import pytest

from krm.phase0.cdx import CdxRecord
from krm.phase0.parsers.hhru import parse_legacy_vacancy, parse_modern_vacancy
from krm.phase0.parsers.linkedin import parse_legacy_linkedin, parse_modern_linkedin
from krm.phase0.schema import (
    SOURCE_TYPES,
    map_hhru_wayback,
    map_linkedin_wayback,
    normalize_record,
    validate_record,
)
from krm.phase0.storage import (
    finish_phase0_run,
    init_phase0_tables,
    start_phase0_run,
    upsert_phase0_vacancy,
)

# ---------------------------------------------------------------------------
# Fixture directory path
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "phase0"


def _load_fixture(name: str) -> str:
    """Load an HTML fixture file as a string."""
    path = FIXTURES_DIR / name
    if not path.exists():
        msg = f"Fixture file not found: {path}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Mock CDX records for each source type
# ---------------------------------------------------------------------------


def _make_cdx_record(
    timestamp: str,
    original: str,
    mimetype: str = "text/html",
    statuscode: str = "200",
    digest: str = "ABC123",
    length: str = "5000",
) -> CdxRecord:
    """Build a CdxRecord from fields."""
    return CdxRecord.from_row([timestamp, original, mimetype, statuscode, digest, length])


PHASE0_FIXTURE_SPECS = [
    pytest.param(
        "hhru_legacy",
        _make_cdx_record(
            "20100315000000",
            "https://web.archive.org/web/20100315000000/https://hh.ru/vacancy.do?id=12345",
        ),
        "hhru_legacy.html",
        "parse_legacy_vacancy",
        "map_hhru_wayback",
        id="hhru_legacy",
    ),
    pytest.param(
        "hhru_modern",
        _make_cdx_record(
            "20231120000000",
            "https://web.archive.org/web/20231120000000/https://hh.ru/vacancy/67890",
        ),
        "hhru_modern.html",
        "parse_modern_vacancy",
        "map_hhru_wayback",
        id="hhru_modern",
    ),
    pytest.param(
        "linkedin_legacy",
        _make_cdx_record(
            "20150601000000",
            "https://web.archive.org/web/20150601000000/https://www.linkedin.com/jobs2/view/12345678",
        ),
        "linkedin_legacy.html",
        "parse_legacy_linkedin",
        "map_linkedin_wayback",
        id="linkedin_legacy",
    ),
    pytest.param(
        "linkedin_modern",
        _make_cdx_record(
            "20240610000000",
            "https://web.archive.org/web/20240610000000/https://www.linkedin.com/jobs/view/ai-scientist-98765432",
        ),
        "linkedin_modern.html",
        "parse_modern_linkedin",
        "map_linkedin_wayback",
        id="linkedin_modern",
    ),
]

# Parser/dispatcher lookup
_PARSERS = {
    "parse_legacy_vacancy": parse_legacy_vacancy,
    "parse_modern_vacancy": parse_modern_vacancy,
    "parse_legacy_linkedin": parse_legacy_linkedin,
    "parse_modern_linkedin": parse_modern_linkedin,
}

_MAPPERS = {
    "map_hhru_wayback": map_hhru_wayback,
    "map_linkedin_wayback": map_linkedin_wayback,
}


# ---------------------------------------------------------------------------
# Unit: CDX → parser end-to-end
# ---------------------------------------------------------------------------


class TestCdxToParsed:
    """Given a mock CdxRecord and fixture HTML, parsing produces structured dicts."""

    def test_fixtures_exist(self) -> None:
        """All 4 fixture files exist on disk."""
        for name in ("hhru_legacy.html", "hhru_modern.html", "linkedin_legacy.html", "linkedin_modern.html"):
            assert (FIXTURES_DIR / name).exists(), f"Missing fixture: {name}"

    @pytest.mark.parametrize("source,cdx_record,fixture_name,parser_name,_mapper_name", PHASE0_FIXTURE_SPECS)
    def test_parse_from_fixture_produces_name(
        self, source: str, cdx_record: CdxRecord, fixture_name: str, parser_name: str, _mapper_name: str
    ) -> None:
        """When: fixture HTML is parsed, Then: name is a non-empty string."""
        html = _load_fixture(fixture_name)
        parser = _PARSERS[parser_name]
        result = parser(html, cdx_record.original)
        assert isinstance(result["name"], str), f"{source}: name should be str, got {type(result['name'])}"
        assert len(result["name"]) > 0, f"{source}: name should not be empty"

    @pytest.mark.parametrize("source,cdx_record,fixture_name,parser_name,_mapper_name", PHASE0_FIXTURE_SPECS)
    def test_parse_from_fixture_produces_parser_version(
        self, source: str, cdx_record: CdxRecord, fixture_name: str, parser_name: str, _mapper_name: str
    ) -> None:
        """When: any fixture HTML is parsed, Then: _parser_version is set."""
        html = _load_fixture(fixture_name)
        parser = _PARSERS[parser_name]
        result = parser(html, cdx_record.original)
        assert "_parser_version" in result, f"{source}: missing _parser_version"
        assert isinstance(result["_parser_version"], str), f"{source}: _parser_version is not str"
        assert len(result["_parser_version"]) > 0, f"{source}: _parser_version is empty"


# ---------------------------------------------------------------------------
# Integration: CDX → parser → mapper → normalize → validate
# ---------------------------------------------------------------------------


class TestCdxToValidated:
    """Given mock CDX + real fixture HTML: parser → mapper → normalize → validate."""

    @pytest.mark.parametrize("source,cdx_record,fixture_name,parser_name,mapper_name", PHASE0_FIXTURE_SPECS)
    def test_full_pipeline_produces_valid_record(
        self, source: str, cdx_record: CdxRecord, fixture_name: str, parser_name: str, mapper_name: str
    ) -> None:
        """When: full pipeline runs, Then: validate_record returns no errors."""
        html = _load_fixture(fixture_name)
        parser = _PARSERS[parser_name]
        mapper = _MAPPERS[mapper_name]

        capture_ts = cdx_record.timestamp
        source_url = cdx_record.original

        # Phase 1: parse HTML
        parsed = parser(html, source_url)

        # Phase 2: map to hh.ru schema
        mapped = mapper(parsed, source_url=source_url, capture_ts=capture_ts)

        # Phase 3: normalize (fill missing fields)
        normalized = normalize_record(mapped)

        # Phase 4: validate
        errors = validate_record(normalized)
        assert errors == [], f"{source}: validation errors: {errors}"

    @pytest.mark.parametrize("source,cdx_record,fixture_name,parser_name,mapper_name", PHASE0_FIXTURE_SPECS)
    def test_mapped_record_has_phase0_metadata(
        self, source: str, cdx_record: CdxRecord, fixture_name: str, parser_name: str, mapper_name: str
    ) -> None:
        """When: record is mapped, Then: _phase0_source, _phase0_capture_ts, _phase0_original_url are set."""
        html = _load_fixture(fixture_name)
        parser = _PARSERS[parser_name]
        mapper = _MAPPERS[mapper_name]

        parsed = parser(html, cdx_record.original)
        mapped = mapper(parsed, source_url=cdx_record.original, capture_ts=cdx_record.timestamp)

        assert mapped["_phase0_source"] in SOURCE_TYPES, f"{source}: invalid source type"
        assert mapped["_phase0_capture_ts"] == cdx_record.timestamp, f"{source}: capture_ts mismatch"
        assert mapped["_phase0_original_url"] == cdx_record.original, f"{source}: original_url mismatch"

    def test_hhru_modern_parses_key_skills(self) -> None:
        """When: hhru_modern fixture is parsed, Then: key_skills list is non-empty with correct shape."""
        html = _load_fixture("hhru_modern.html")
        parsed = parse_modern_vacancy(html, "https://hh.ru/vacancy/67890")
        skills = parsed.get("key_skills", [])
        assert len(skills) > 0, "Modern hh.ru fixture should have key_skills"
        for skill in skills:
            assert set(skill.keys()) == {"name"}, f"Unexpected keys in skill: {skill.keys()}"

    def test_linkedin_modern_parses_salary_from_jsonld(self) -> None:
        """When: linkedin_modern fixture with JSON-LD is parsed, Then: salary range is extracted."""
        html = _load_fixture("linkedin_modern.html")
        parsed = parse_modern_linkedin(html, "https://linkedin.com/jobs/view/98765432")
        salary = parsed.get("salary", {})
        assert salary.get("from") == 180000
        assert salary.get("to") == 250000
        assert salary.get("currency") == "USD"

    def test_linkedin_legacy_parses_description_and_criteria(self) -> None:
        """When: linkedin_legacy fixture is parsed, Then: description, experience, industry are set."""
        html = _load_fixture("linkedin_legacy.html")
        parsed = parse_legacy_linkedin(html, "https://linkedin.com/jobs2/view/12345678")
        assert parsed["description"] is not None, "description should not be None"
        assert "Python" in parsed["description"]
        assert parsed["experience"]["name"] == "Mid-Senior level"
        assert parsed["industry"] == "Information Technology and Services"


# ---------------------------------------------------------------------------
# E2E: CDX → parser → mapper → storage (DuckDB)
# ---------------------------------------------------------------------------


class TestE2EWithStorage:
    """Full pipeline with real DuckDB storage: parse → map → normalize → upsert."""

    @pytest.mark.parametrize("source,cdx_record,fixture_name,parser_name,mapper_name", PHASE0_FIXTURE_SPECS)
    def test_store_single_vacancy_and_retrieve(
        self, source: str, cdx_record: CdxRecord, fixture_name: str, parser_name: str, mapper_name: str
    ) -> None:
        """When: a single vacancy goes through full pipeline to DuckDB, Then: it is retrievable."""
        html = _load_fixture(fixture_name)
        parser = _PARSERS[parser_name]
        mapper = _MAPPERS[mapper_name]

        parsed = parser(html, cdx_record.original)
        mapped = mapper(parsed, source_url=cdx_record.original, capture_ts=cdx_record.timestamp)
        normalized = normalize_record(mapped)
        errors = validate_record(normalized)
        assert errors == [], f"{source}: {errors}"

        # Store in temp DuckDB
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            run_id = f"test-run-{source}"
            vacancy_id = normalized["id"]

            start_phase0_run(conn, run_id, source)
            upsert_phase0_vacancy(conn, run_id, vacancy_id, normalized)
            finish_phase0_run(conn, run_id, records_fetched=1, records_stored=1)

            # Verify run was recorded
            run_row = conn.execute(
                "SELECT source, records_fetched, records_stored FROM phase0_runs WHERE run_id = ?",
                [run_id],
            ).fetchone()
            assert run_row is not None, f"{source}: run not found"
            assert run_row[1] == 1  # records_fetched
            assert run_row[2] == 1  # records_stored

            # Verify vacancy is stored
            stored = conn.execute(
                "SELECT id, run_id, data FROM raw_vacancies WHERE id = ?",
                [vacancy_id],
            ).fetchone()
            assert stored is not None, f"{source}: vacancy not found"
            assert stored[0] == vacancy_id
            assert stored[1] == run_id

            retrieved = json.loads(stored[2])
            assert retrieved["name"] == normalized["name"]
            assert retrieved["_phase0_source"] == normalized["_phase0_source"]

        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_store_all_four_sources_in_single_db(self) -> None:
        """When: all 4 source types are stored in the same DB, Then: 4 vacancies are counted."""
        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            total_stored = 0
            for spec in PHASE0_FIXTURE_SPECS:
                source, cdx_record, fixture_name, parser_name, mapper_name = spec.values
                html = _load_fixture(fixture_name)
                parser = _PARSERS[parser_name]
                mapper = _MAPPERS[mapper_name]

                parsed = parser(html, cdx_record.original)
                mapped = mapper(parsed, source_url=cdx_record.original, capture_ts=cdx_record.timestamp)
                normalized = normalize_record(mapped)
                errors = validate_record(normalized)
                if errors:
                    continue

                run_id = f"multi-{source}"
                start_phase0_run(conn, run_id, source)
                upsert_phase0_vacancy(conn, run_id, normalized["id"], normalized)
                finish_phase0_run(conn, run_id, records_fetched=1, records_stored=1)
                total_stored += 1

            # Verify counts
            run_count = conn.execute("SELECT COUNT(*) FROM phase0_runs").fetchone()[0]
            vac_count = conn.execute("SELECT COUNT(*) FROM raw_vacancies").fetchone()[0]
            assert run_count == 4, f"Expected 4 runs, got {run_count}"
            assert vac_count == 4, f"Expected 4 vacancies, got {vac_count}"

            # Verify all 4 runs exist (by source labels: hhru_legacy, hhru_modern, linkedin_legacy, linkedin_modern)
            sources = conn.execute("SELECT DISTINCT source FROM phase0_runs ORDER BY source").fetchall()
            source_names = {row[0] for row in sources}
            assert len(source_names) == 4, f"Expected 4 distinct run sources, got {source_names}"

            # Each vacancy has _phase0_source in its JSON data — 2 distinct mapper sources
            rows = conn.execute("SELECT data FROM raw_vacancies").fetchall()
            stored_sources = {json.loads(row[0]).get("_phase0_source") for row in rows}
            assert len(stored_sources) == 2, f"Expected 2 distinct _phase0_source values, got {stored_sources}"
            assert stored_sources == {"wayback-hhru", "wayback-linkedin"}

        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)

    def test_duplicate_id_is_ignored(self) -> None:
        """When: a vacancy with the same id is inserted twice, Then: only one row exists."""
        _, cdx_record, fixture_name, parser_name, mapper_name = PHASE0_FIXTURE_SPECS[0].values

        html = _load_fixture(fixture_name)
        parser = _PARSERS[parser_name]
        mapper = _MAPPERS[mapper_name]

        parsed = parser(html, cdx_record.original)
        mapped = mapper(parsed, source_url=cdx_record.original, capture_ts=cdx_record.timestamp)
        normalized = normalize_record(mapped)

        db_path = None
        conn = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                db_path = f.name
            Path(db_path).unlink(missing_ok=True)
            conn = duckdb.connect(db_path)
            init_phase0_tables(conn)

            start_phase0_run(conn, "run-1", "hhru_legacy")
            upsert_phase0_vacancy(conn, "run-1", normalized["id"], normalized)

            start_phase0_run(conn, "run-2", "hhru_legacy")
            upsert_phase0_vacancy(conn, "run-2", normalized["id"], normalized)

            count = conn.execute("SELECT COUNT(*) FROM raw_vacancies").fetchone()[0]
            assert count == 1, f"Expected 1 vacancy, got {count}"

        finally:
            if conn is not None:
                conn.close()
            if db_path is not None:
                Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E2E: Error/edge cases
# ---------------------------------------------------------------------------


class TestE2EEdgeCases:
    """Edge cases across the full pipeline."""

    def test_empty_html_name_is_empty_raises_on_normalize(self) -> None:
        """When: empty HTML is parsed and mapped, Then: normalize_record raises ValueError (empty name)."""
        cdx = _make_cdx_record("20240101000000", "https://hh.ru/vacancy/99999")
        parsed = parse_modern_vacancy("<html><body></body></html>", cdx.original)
        mapped = map_hhru_wayback(parsed, source_url=cdx.original, capture_ts=cdx.timestamp)
        # normalize_record requires "name" to be non-empty
        with pytest.raises(ValueError, match="name"):
            normalize_record(mapped)

    def test_login_gated_linkedin_legacy_produces_null_name(self) -> None:
        """When: login-gated LinkedIn legacy page is parsed, Then: name is None."""
        html = """<html><body>
        <div class="sign-in-modal"><h2>Sign in to view this job</h2></div>
        <form class="login"></form>
        </body></html>"""
        cdx = _make_cdx_record("20240101000000", "https://linkedin.com/jobs2/view/12345")
        parsed = parse_legacy_linkedin(html, cdx.original)
        assert parsed["name"] is None, "Login-gated page should return None name"

    def test_vacancy_id_from_url_when_not_in_parsed_data(self) -> None:
        """When: parsed data has no 'id' field, Then: mapper extracts id from URL."""
        cdx = _make_cdx_record("20240101000000", "https://hh.ru/vacancy/55555")
        parsed = {"name": "Some Job"}  # no 'id' key
        mapped = map_hhru_wayback(parsed, source_url=cdx.original, capture_ts=cdx.timestamp)
        assert mapped["id"] == "55555"
