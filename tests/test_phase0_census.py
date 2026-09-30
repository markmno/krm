"""Tests for phase0/census.py — CDX census coverage estimation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from krm.phase0.cdx import CdxQuerier, CdxRecord
from krm.phase0.census import CensusReport, CensusResult, CensusRunner


# ======================================================================
# Helpers
# ======================================================================


def _make_record(original: str, timestamp: str = "20240101000000") -> CdxRecord:
    """Create a single CdxRecord for test data."""
    return CdxRecord(
        urlkey="",
        timestamp=timestamp,
        original=original,
        mimetype="text/html",
        statuscode="200",
        digest="ABC123",
        length="1000",
    )


def _make_capture_records(count: int, base_url: str) -> list[CdxRecord]:
    """Return *count* records with sequentially numbered URLs."""
    return [
        _make_record(f"http://{base_url}/{i}", f"202401010{i:05d}")
        for i in range(count)
    ]


def _make_unique_records(count: int, base_url: str) -> list[CdxRecord]:
    """Return *count* records representing unique URLs (collapsed)."""
    return [
        _make_record(f"http://{base_url}/unique/{i}")
        for i in range(count)
    ]


# ======================================================================
# CensusResult
# ======================================================================


class TestCensusResult:
    """Tests for CensusResult dataclass creation and properties."""

    def test_creation_and_field_access(self) -> None:
        """All fields are accessible after construction."""
        result = CensusResult(
            source="wayback-hhru",
            url_pattern="hh.ru/vacancy/*",
            year=2020,
            capture_count=5000,
            unique_urls=1200,
        )

        assert result.source == "wayback-hhru"
        assert result.url_pattern == "hh.ru/vacancy/*"
        assert result.year == 2020
        assert result.capture_count == 5000
        assert result.unique_urls == 1200

    def test_is_frozen(self) -> None:
        """CensusResult is immutable."""
        result = CensusResult(
            source="wayback-hhru",
            url_pattern="hh.ru/vacancy/*",
            year=2020,
            capture_count=100,
            unique_urls=50,
        )

        with pytest.raises(Exception):
            result.capture_count = 999  # pyright: ignore[reportAttributeAccessIssue]

    def test_different_sources_have_different_labels(self) -> None:
        """Source label distinguishes between different data origins."""
        hhru = CensusResult(
            source="wayback-hhru",
            url_pattern="hh.ru/vacancy/*",
            year=2020,
            capture_count=100,
            unique_urls=50,
        )
        linkedin = CensusResult(
            source="wayback-linkedin",
            url_pattern="linkedin.com/jobs/view/*",
            year=2020,
            capture_count=100,
            unique_urls=50,
        )

        assert hhru.source != linkedin.source


# ======================================================================
# CensusReport.to_json()
# ======================================================================


class TestCensusReportToJson:
    """Tests for CensusReport JSON serialization."""

    def test_writes_valid_json_with_correct_structure(self) -> None:
        """to_json() produces parseable JSON with top-level keys."""
        report = CensusReport(
            generated_at="2024-01-01T00:00:00+00:00",
            results=[
                CensusResult(
                    source="wayback-hhru",
                    url_pattern="hh.ru/vacancy/*",
                    year=2020,
                    capture_count=100,
                    unique_urls=50,
                ),
                CensusResult(
                    source="wayback-hhru",
                    url_pattern="hh.ru/vacancy/*",
                    year=2021,
                    capture_count=200,
                    unique_urls=80,
                ),
            ],
            total_estimated_records=300,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "census.json"
            report.to_json(path)

            assert path.exists()
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

        assert data["generated_at"] == "2024-01-01T00:00:00+00:00"
        assert data["total_estimated_records"] == 300
        assert len(data["results"]) == 2
        assert data["results"][0]["source"] == "wayback-hhru"
        assert data["results"][0]["year"] == 2020
        assert data["results"][0]["capture_count"] == 100
        assert data["results"][0]["unique_urls"] == 50

    def test_creates_parent_directories(self) -> None:
        """to_json() creates missing parent directories."""
        report = CensusReport(
            generated_at="2024-01-01T00:00:00+00:00",
            results=[],
            total_estimated_records=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "census.json"
            report.to_json(nested)

            assert nested.exists()

    def test_empty_report_writes_empty_results_array(self) -> None:
        """A report with no results writes an empty list."""
        report = CensusReport(
            generated_at="2024-01-01T00:00:00+00:00",
            results=[],
            total_estimated_records=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            report.to_json(path)

            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

        assert data["results"] == []
        assert data["total_estimated_records"] == 0


# ======================================================================
# CensusReport.to_table()
# ======================================================================


class TestCensusReportToTable:
    """Tests for the ASCII table output."""

    def test_produces_non_empty_output(self) -> None:
        """to_table() returns a string with expected content."""
        report = CensusReport(
            generated_at="2024-01-01T00:00:00+00:00",
            results=[
                CensusResult(
                    source="wayback-hhru",
                    url_pattern="hh.ru/vacancy/*",
                    year=2020,
                    capture_count=100,
                    unique_urls=50,
                ),
            ],
            total_estimated_records=100,
        )

        table = report.to_table()

        assert "Census Report" in table
        assert "2024-01-01" in table
        assert "100" in table
        assert "wayback-hhru" in table
        assert "2020" in table
        assert "50" in table

    def test_includes_header_row(self) -> None:
        """Table includes a header with Source, Pattern, Year, Captures, Unique."""
        report = CensusReport(
            generated_at="2024-01-01T00:00:00+00:00",
            results=[],
            total_estimated_records=0,
        )

        table = report.to_table()

        assert "Source" in table
        assert "Pattern" in table
        assert "Year" in table
        assert "Captures" in table
        assert "Unique" in table

    def test_formats_numbers_with_commas(self) -> None:
        """Large numbers are formatted with thousands separators."""
        report = CensusReport(
            generated_at="2024-01-01T00:00:00+00:00",
            results=[
                CensusResult(
                    source="wayback-hhru",
                    url_pattern="hh.ru/vacancy/*",
                    year=2020,
                    capture_count=12345,
                    unique_urls=6789,
                ),
            ],
            total_estimated_records=12345,
        )

        table = report.to_table()

        assert "12,345" in table
        assert "6,789" in table


# ======================================================================
# CensusReport JSON round-trip
# ======================================================================


class TestCensusReportRoundTrip:
    """JSON write-then-read round-trip tests."""

    def test_json_round_trip_preserves_all_data(self) -> None:
        """A report written to JSON can be read back with all fields intact."""
        original = CensusReport(
            generated_at="2024-06-15T12:00:00+00:00",
            results=[
                CensusResult(
                    source="wayback-hhru",
                    url_pattern="hh.ru/vacancy/*",
                    year=2020,
                    capture_count=500,
                    unique_urls=200,
                ),
                CensusResult(
                    source="wayback-linkedin",
                    url_pattern="linkedin.com/jobs/view/*",
                    year=2022,
                    capture_count=300,
                    unique_urls=150,
                ),
            ],
            total_estimated_records=800,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.json"
            original.to_json(path)

            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

        assert data["generated_at"] == original.generated_at
        assert data["total_estimated_records"] == original.total_estimated_records
        assert len(data["results"]) == len(original.results)

        for i, orig_result in enumerate(original.results):
            rd = data["results"][i]
            assert rd["source"] == orig_result.source
            assert rd["url_pattern"] == orig_result.url_pattern
            assert rd["year"] == orig_result.year
            assert rd["capture_count"] == orig_result.capture_count
            assert rd["unique_urls"] == orig_result.unique_urls


# ======================================================================
# CensusRunner — mock CdxQuerier.query
# ======================================================================


def _mock_query_side_effect(
    capture_count: int,
    unique_count: int,
    base_url: str,
) -> object:
    """Return a side_effect function for CdxQuerier.query that returns
    controlled capture and unique record lists depending on the ``collapse``
    kwarg.
    """

    def side_effect(
        url_pattern: str,
        *,
        from_ts: str | None = None,
        to_ts: str | None = None,
        collapse: str | None = None,
        match_type: str | None = None,
        status_filter: int | None = 200,
        limit: int | None = None,
    ) -> list[CdxRecord]:
        if collapse == "urlkey":
            return _make_unique_records(unique_count, base_url)
        return _make_capture_records(capture_count, base_url)

    return side_effect


# ======================================================================
# CensusRunner.run_hhru_wayback()
# ======================================================================


class TestRunHhruWayback:
    """Tests for CensusRunner.run_hhru_wayback()."""

    def test_returns_correct_number_of_results(self) -> None:
        """Returns 18 results: 2 legacy (2010-2011) + 16 modern (2010-2025)."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=100,
            unique_count=50,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_hhru_wayback()

        # 2 legacy years + 16 modern years = 18 total
        assert len(results) == 18

    def test_includes_legacy_and_modern_patterns(self) -> None:
        """Results include both legacy (hh.ru/vacancy*.do) and modern patterns."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=100,
            unique_count=50,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_hhru_wayback()

        patterns = {r.url_pattern for r in results}
        assert "hh.ru/vacancy*.do" in patterns
        assert "hh.ru/vacancy/*" in patterns

    def test_legacy_pattern_only_for_2010_2011(self) -> None:
        """Legacy pattern is queried only for years 2010 and 2011."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=100,
            unique_count=50,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_hhru_wayback()

        legacy = [r for r in results if r.url_pattern == "hh.ru/vacancy*.do"]
        legacy_years = {r.year for r in legacy}

        assert legacy_years == {2010, 2011}
        assert len(legacy) == 2

    def test_modern_pattern_covers_2010_thru_2025(self) -> None:
        """Modern pattern covers all years from 2010 through 2025."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=100,
            unique_count=50,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_hhru_wayback()

        modern = [r for r in results if r.url_pattern == "hh.ru/vacancy/*"]
        modern_years = {r.year for r in modern}

        assert modern_years == set(range(2010, 2026))
        assert len(modern) == 16

    def test_queries_pass_correct_from_and_to_timestamps(self) -> None:
        """Each query passes year boundaries as from_ts/to_ts."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=1,
            unique_count=1,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect) as mock_query:
            runner = CensusRunner(querier)
            runner.run_hhru_wayback()

        # Check the first modern-pattern call (year 2010).
        modern_2010_calls = [
            c for c in mock_query.call_args_list
            if c.kwargs.get("url_pattern") == "hh.ru/vacancy/*"
            and not c.kwargs.get("collapse")
        ]
        assert len(modern_2010_calls) >= 1
        call = modern_2010_calls[0]
        assert call.kwargs["from_ts"] == "20100101"
        assert call.kwargs["to_ts"] == "20101231"

    def test_uses_collapse_urlkey_for_unique_count(self) -> None:
        """Queries with collapse='urlkey' are issued for each year."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=100,
            unique_count=50,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect) as mock_query:
            runner = CensusRunner(querier)
            runner.run_hhru_wayback()

        collapsed_calls = [
            c for c in mock_query.call_args_list
            if c.kwargs.get("collapse") == "urlkey"
        ]
        # 18 year×pattern combos, each with a collapse query
        assert len(collapsed_calls) == 18

    def test_uses_status_filter_200(self) -> None:
        """All queries filter to HTTP 200 status codes."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=1,
            unique_count=1,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect) as mock_query:
            runner = CensusRunner(querier)
            runner.run_hhru_wayback()

        for call in mock_query.call_args_list:
            assert call.kwargs["status_filter"] == 200

    def test_capture_and_unique_counts_are_recorded(self) -> None:
        """Each CensusResult stores capture_count and unique_urls correctly."""
        querier = CdxQuerier()
        # Return 200 captures and 80 unique URLs for every query.
        side_effect = _mock_query_side_effect(
            capture_count=200,
            unique_count=80,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_hhru_wayback()

        for r in results:
            assert r.capture_count == 200
            assert r.unique_urls == 80

    def test_all_results_have_wayback_hhru_source(self) -> None:
        """All hhru results are labeled 'wayback-hhru'."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=10,
            unique_count=5,
            base_url="hh.ru",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_hhru_wayback()

        for r in results:
            assert r.source == "wayback-hhru"


# ======================================================================
# CensusRunner.run_linkedin_wayback()
# ======================================================================


class TestRunLinkedinWayback:
    """Tests for CensusRunner.run_linkedin_wayback()."""

    def test_returns_correct_number_of_results(self) -> None:
        """Returns 12 results: 4 legacy (2013-2016) + 8 modern (2019-2026)."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=50,
            unique_count=25,
            base_url="linkedin.com",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_linkedin_wayback()

        assert len(results) == 12

    def test_legacy_pattern_uses_jobs2_view(self) -> None:
        """Legacy LinkedIn pattern targets linkedin.com/jobs2/view/*."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=50,
            unique_count=25,
            base_url="linkedin.com",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_linkedin_wayback()

        legacy = [r for r in results if r.url_pattern == "linkedin.com/jobs2/view/*"]
        assert len(legacy) == 4
        assert {r.year for r in legacy} == {2013, 2014, 2015, 2016}

    def test_modern_pattern_uses_jobs_view(self) -> None:
        """Modern LinkedIn pattern targets linkedin.com/jobs/view/*."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=50,
            unique_count=25,
            base_url="linkedin.com",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_linkedin_wayback()

        modern = [r for r in results if r.url_pattern == "linkedin.com/jobs/view/*"]
        assert len(modern) == 8
        assert {r.year for r in modern} == set(range(2019, 2027))

    def test_all_results_have_wayback_linkedin_source(self) -> None:
        """All LinkedIn results are labeled 'wayback-linkedin'."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=10,
            unique_count=5,
            base_url="linkedin.com",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            results = runner.run_linkedin_wayback()

        for r in results:
            assert r.source == "wayback-linkedin"


# ======================================================================
# CensusRunner.run_all()
# ======================================================================


class TestRunAll:
    """Tests for CensusRunner.run_all() — aggregation and report generation."""

    def test_combines_hhru_and_linkedin_results(self) -> None:
        """run_all() includes results from both hhru and linkedin sources."""
        querier = CdxQuerier()

        # Return different counts per source to verify aggregation.
        call_count = 0

        def side_effect(
            url_pattern: str,
            *,
            from_ts: str | None = None,
            to_ts: str | None = None,
            collapse: str | None = None,
            match_type: str | None = None,
            status_filter: int | None = 200,
            limit: int | None = None,
        ) -> list[CdxRecord]:
            nonlocal call_count
            call_count += 1
            if collapse == "urlkey":
                return _make_unique_records(30, "test.example")
            return _make_capture_records(50, "test.example")

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            report = runner.run_all()

        sources = {r.source for r in report.results}
        assert "wayback-hhru" in sources
        assert "wayback-linkedin" in sources

        # 18 hhru results + 12 linkedin = 30 total
        assert len(report.results) == 30

    def test_computes_total_estimated_records_correctly(self) -> None:
        """total_estimated_records equals sum of all capture_counts."""
        querier = CdxQuerier()

        # Each query returns 2 capture records and 1 unique record.
        # Total queries: 30 year×pattern combos × 2 queries each = 60 queries.
        # Each capture query returns 2 records.
        # Total capture_count sum: 30 results × 2 = 60.
        def side_effect(
            url_pattern: str,
            *,
            from_ts: str | None = None,
            to_ts: str | None = None,
            collapse: str | None = None,
            match_type: str | None = None,
            status_filter: int | None = 200,
            limit: int | None = None,
        ) -> list[CdxRecord]:
            if collapse == "urlkey":
                return _make_unique_records(3, "test.example")
            return _make_capture_records(2, "test.example")

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            report = runner.run_all()

        # Sum of capture_count across all results: 30 results × 2 = 60
        assert report.total_estimated_records == 60

    def test_generated_at_is_iso_timestamp(self) -> None:
        """generated_at contains an ISO-8601 timestamp."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=1,
            unique_count=1,
            base_url="test.example",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            report = runner.run_all()

        # ISO-8601: contains 'T' separator and timezone info
        assert "T" in report.generated_at
        assert "+" in report.generated_at or "Z" in report.generated_at
        # Should be parseable by datetime
        from datetime import datetime
        datetime.fromisoformat(report.generated_at)

    def test_report_includes_all_census_result_fields(self) -> None:
        """Each result in the report has all required fields populated."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=42,
            unique_count=17,
            base_url="test.example",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            report = runner.run_all()

        for r in report.results:
            assert r.source in ("wayback-hhru", "wayback-linkedin")
            assert r.url_pattern != ""
            assert 2010 <= r.year <= 2026
            assert r.capture_count > 0
            assert r.unique_urls > 0

    def test_json_output_is_reproducible(self) -> None:
        """The report can be serialized and deserialized identically."""
        querier = CdxQuerier()
        side_effect = _mock_query_side_effect(
            capture_count=1,
            unique_count=1,
            base_url="test.example",
        )

        with mock.patch.object(querier, "query", side_effect=side_effect):
            runner = CensusRunner(querier)
            report = runner.run_all()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "census.json"
            report.to_json(path)

            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

        assert data["total_estimated_records"] == report.total_estimated_records
        assert len(data["results"]) == len(report.results)
        for orig, loaded in zip(report.results, data["results"]):
            assert loaded["source"] == orig.source
            assert loaded["year"] == orig.year
            assert loaded["capture_count"] == orig.capture_count
            assert loaded["unique_urls"] == orig.unique_urls
