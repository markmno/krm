"""Tests for LinkedIn Wayback Machine collector.

All CDX API and Wayback Machine calls are mocked — no real HTTP requests.
"""

from __future__ import annotations

import logging
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from krm.phase0.cdx import CdxRecord
from krm.phase0.collectors.linkedin_wayback import collect_linkedin_wayback
from krm.phase0.fetcher import FetchedPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cdx_record(
    original: str = "https://linkedin.com/jobs/view/12345",
    timestamp: str = "20200101120000",
) -> CdxRecord:
    """Create a minimal CdxRecord for testing."""
    return CdxRecord(
        urlkey="",
        timestamp=timestamp,
        original=original,
        mimetype="text/html",
        statuscode="200",
        digest="abc123",
        length="5000",
    )


def _make_fetched_page(
    cdx_record: CdxRecord,
    html: str = "<html><body>Job</body></html>",
) -> FetchedPage:
    """Create a minimal FetchedPage for testing."""
    return FetchedPage(
        cdx_record=cdx_record,
        html_content=html,
        content_type="text/html",
        fetched_at="2025-01-01T00:00:00Z",
    )


def _make_parsed() -> dict[str, object]:
    """Create a minimal successful parse result."""
    return {
        "name": "Software Engineer",
        "description": "Build things",
        "employer": {"name": "Acme Corp"},
        "area": {"name": "San Francisco"},
        "salary": {"from": None, "to": None, "currency": None},
        "published_at": None,
        "key_skills": ["Python", "Go"],
        "experience": {"name": "Senior"},
        "industry": "Technology",
        "professional_roles": [],
        "_parser_version": "linkedin_modern_v1",
        "_source_url": "https://linkedin.com/jobs/view/12345",
    }


# ---------------------------------------------------------------------------
# Tests: Era detection → correct parser
# ---------------------------------------------------------------------------


class TestEraDetection:
    """collect_linkedin_wayback selects correct URL pattern and parser by year."""

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_legacy_era_uses_parse_legacy_linkedin(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given year=2015 (legacy era) and a /jobs2/view/ URL,
        When collect_linkedin_wayback runs,
        Then parse_legacy_linkedin is called and parse_modern_linkedin is not.
        """
        cdx_rec = _make_cdx_record(
            original="https://linkedin.com/jobs2/view/12345",
            timestamp="20150615120000",
        )
        page = _make_fetched_page(cdx_rec, "<html>legacy job</html>")
        parsed = {
            **_make_parsed(),
            "_parser_version": "linkedin_legacy_v1",
            "_source_url": cdx_rec.original,
        }

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([cdx_rec])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.return_value = page
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_legacy_linkedin",
            return_value=parsed,
        ) as mock_legacy:
            with patch(
                "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin"
            ) as mock_modern:
                result = collect_linkedin_wayback(2015)

        assert result == 1
        mock_legacy.assert_called_once_with(page.html_content, cdx_rec.original)
        mock_modern.assert_not_called()

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_modern_era_uses_parse_modern_linkedin(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given year=2020 (modern era) and a /jobs/view/ URL,
        When collect_linkedin_wayback runs,
        Then parse_modern_linkedin is called and parse_legacy_linkedin is not.
        """
        cdx_rec = _make_cdx_record(
            original="https://linkedin.com/jobs/view/slug-98765",
            timestamp="20200315120000",
        )
        page = _make_fetched_page(cdx_rec, "<html>modern job</html>")
        parsed = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([cdx_rec])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.return_value = page
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            return_value=parsed,
        ) as mock_modern:
            with patch(
                "krm.phase0.collectors.linkedin_wayback.parse_legacy_linkedin"
            ) as mock_legacy:
                result = collect_linkedin_wayback(2020)

        assert result == 1
        mock_modern.assert_called_once_with(page.html_content, cdx_rec.original)
        mock_legacy.assert_not_called()

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_modern_url_uses_legacy_parser_when_url_has_jobs2(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given year=2020 but the CDX URL contains /jobs2/view/,
        When collect_linkedin_wayback runs,
        Then parse_legacy_linkedin is called (URL-based detection, not year-based).
        """
        cdx_rec = _make_cdx_record(
            original="https://linkedin.com/jobs2/view/12345",
            timestamp="20200315120000",
        )
        page = _make_fetched_page(cdx_rec, "<html>legacy job in modern year</html>")
        parsed = {
            **_make_parsed(),
            "_parser_version": "linkedin_legacy_v1",
            "_source_url": cdx_rec.original,
        }

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([cdx_rec])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.return_value = page
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_legacy_linkedin",
            return_value=parsed,
        ) as mock_legacy:
            with patch(
                "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin"
            ) as mock_modern:
                result = collect_linkedin_wayback(2020)

        assert result == 1
        mock_legacy.assert_called_once()
        mock_modern.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: 2017–2018 skip
# ---------------------------------------------------------------------------


class TestLinkedInSkipYears:
    """collect_linkedin_wayback returns 0 for 2017 and 2018 with a log message."""

    def test_linkedin_2017_returns_zero(self) -> None:
        """Given year=2017, When collect_linkedin_wayback, Then return 0."""
        result = collect_linkedin_wayback(2017)
        assert result == 0

    def test_linkedin_2018_returns_zero(self) -> None:
        """Given year=2018, When collect_linkedin_wayback, Then return 0."""
        result = collect_linkedin_wayback(2018)
        assert result == 0

    def test_linkedin_2017_logs_skip_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Given year=2017, When collect_linkedin_wayback,
        Then a skip log message is emitted at INFO level.
        """
        with caplog.at_level(logging.INFO, logger="krm.phase0.collectors.linkedin_wayback"):
            collect_linkedin_wayback(2017)

        assert "No LinkedIn Wayback coverage in 2017" in caplog.text

    def test_linkedin_2018_logs_skip_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Given year=2018, When collect_linkedin_wayback,
        Then a skip log message is emitted.
        """
        with caplog.at_level(logging.INFO, logger="krm.phase0.collectors.linkedin_wayback"):
            collect_linkedin_wayback(2018)

        assert "No LinkedIn Wayback coverage in 2018" in caplog.text


# ---------------------------------------------------------------------------
# Tests: Storage integration
# ---------------------------------------------------------------------------


class TestLinkedInStorageIntegration:
    """collect_linkedin_wayback correctly interacts with the storage layer."""

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_upsert_called_with_year_run_id(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given year=2021 and one successfully parsed vacancy,
        When collect_linkedin_wayback runs,
        Then upsert_phase0_vacancy is called with run_id='linkedin-2021'.
        """
        cdx_rec = _make_cdx_record(
            original="https://linkedin.com/jobs/view/12345",
        )
        page = _make_fetched_page(cdx_rec)
        parsed = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([cdx_rec])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.return_value = page
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            return_value=parsed,
        ):
            result = collect_linkedin_wayback(2021)

        assert result == 1
        mock_start.assert_called_once_with(
            mock_conn,
            run_id="linkedin-2021",
            source="linkedin_wayback",
            url_pattern="linkedin.com/jobs/view/*",
            params={"year": 2021},
        )
        mock_upsert.assert_called_once()
        _, kwargs = mock_upsert.call_args_list[0]
        _args = mock_upsert.call_args_list[0][0]
        assert _args[1] == "linkedin-2021"  # run_id

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_finish_run_called_with_correct_counts(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given two CDX records, both parsed successfully,
        When collect_linkedin_wayback runs,
        Then finish_phase0_run is called with fetched=2, stored=2.
        """
        rec1 = _make_cdx_record(
            original="https://linkedin.com/jobs/view/111",
            timestamp="20190101120000",
        )
        rec2 = _make_cdx_record(
            original="https://linkedin.com/jobs/view/222",
            timestamp="20190201120000",
        )
        page1 = _make_fetched_page(rec1)
        page2 = _make_fetched_page(rec2)
        parsed = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([rec1, rec2])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.side_effect = [page1, page2]
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            return_value=parsed,
        ):
            result = collect_linkedin_wayback(2019)

        assert result == 2
        mock_finish.assert_called_once_with(
            mock_conn, "linkedin-2019", 2, 2
        )

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_init_tables_called(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given any year with results, When collect_linkedin_wayback runs,
        Then init_phase0_tables is called with the connection.
        """
        cdx_rec = _make_cdx_record()
        page = _make_fetched_page(cdx_rec)
        parsed = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([cdx_rec])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.return_value = page
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            return_value=parsed,
        ):
            collect_linkedin_wayback(2020)

        mock_init.assert_called_once_with(mock_conn)


# ---------------------------------------------------------------------------
# Tests: Parse failure → skip gracefully
# ---------------------------------------------------------------------------


class TestLinkedInParseFailures:
    """Parse failures are skipped gracefully without crashing the pipeline."""

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_parse_exception_skipped(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given a CDX record whose parsing raises an exception,
        When collect_linkedin_wayback runs,
        Then the exception is caught, the record is skipped, and the pipeline continues.
        """
        rec1 = _make_cdx_record(original="https://linkedin.com/jobs/view/bad")
        rec2 = _make_cdx_record(original="https://linkedin.com/jobs/view/good")
        page1 = _make_fetched_page(rec1, "<html>broken</html>")
        page2 = _make_fetched_page(rec2, "<html>valid</html>")
        parsed2 = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([rec1, rec2])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.side_effect = [page1, page2]
        mock_fetcher_cls.return_value = mock_fetcher

        def _parse_side_effect(html: str, url: str) -> dict[str, object]:
            if "bad" in url:
                raise ValueError("Parse error")
            return parsed2

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            side_effect=_parse_side_effect,
        ):
            result = collect_linkedin_wayback(2020)

        assert result == 1  # Only the good record stored
        mock_upsert.assert_called_once()
        mock_finish.assert_called_once_with(mock_conn, "linkedin-2020", 2, 1)

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_normalize_value_error_skipped(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given a parse result that fails normalize_record (e.g. missing name),
        When collect_linkedin_wayback runs,
        Then the record is skipped and the pipeline continues.
        """
        rec1 = _make_cdx_record(original="https://linkedin.com/jobs/view/bad-name")
        rec2 = _make_cdx_record(original="https://linkedin.com/jobs/view/good")
        page1 = _make_fetched_page(rec1, "<html>no name</html>")
        page2 = _make_fetched_page(rec2, "<html>valid</html>")

        parsed1: dict[str, object] = {"name": None, "_source_url": rec1.original}
        parsed2 = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([rec1, rec2])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.side_effect = [page1, page2]
        mock_fetcher_cls.return_value = mock_fetcher

        def _parse_side_effect(html: str, url: str) -> dict[str, object]:
            if "bad-name" in url:
                return parsed1
            return parsed2

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            side_effect=_parse_side_effect,
        ):
            result = collect_linkedin_wayback(2020)

        assert result == 1  # Only the good record stored
        mock_upsert.assert_called_once()

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_fetch_returns_none_skipped(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given a CDX record whose fetch returns None (e.g. 404),
        When collect_linkedin_wayback runs,
        Then the record is skipped and no parse or store is attempted.
        """
        rec1 = _make_cdx_record(original="https://linkedin.com/jobs/view/missing")
        rec2 = _make_cdx_record(original="https://linkedin.com/jobs/view/good")
        page2 = _make_fetched_page(rec2, "<html>valid</html>")
        parsed2 = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([rec1, rec2])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.side_effect = [None, page2]
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            return_value=parsed2,
        ) as mock_parse:
            result = collect_linkedin_wayback(2020)

        assert result == 1
        mock_parse.assert_called_once()  # Only parsed once for the good record


# ---------------------------------------------------------------------------
# Tests: Empty CDX results
# ---------------------------------------------------------------------------


class TestLinkedInEmptyCdxResults:
    """Empty CDX results → stored=0, finish_run still called."""

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_empty_cdx_returns_zero(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given a year with no CDX results,
        When collect_linkedin_wayback runs,
        Then the function returns 0 and upsert is never called.
        """
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([])  # Empty
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher

        result = collect_linkedin_wayback(2020)

        assert result == 0
        mock_upsert.assert_not_called()
        mock_finish.assert_called_once_with(mock_conn, "linkedin-2020", 0, 0)

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_empty_cdx_still_finishes_run(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given empty CDX results, When collect_linkedin_wayback runs,
        Then finish_phase0_run is still called with fetched=0, stored=0.
        """
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter([])
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher

        collect_linkedin_wayback(2019)

        mock_start.assert_called_once()
        mock_finish.assert_called_once_with(mock_conn, "linkedin-2019", 0, 0)


# ---------------------------------------------------------------------------
# Tests: Multiple records — counts
# ---------------------------------------------------------------------------


class TestLinkedInMultipleRecords:
    """Pipeline processes multiple records correctly."""

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_three_successful_records(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given 3 CDX records, all parsed successfully,
        When collect_linkedin_wayback runs,
        Then result=3 and all three are upserted.
        """
        records = [
            _make_cdx_record(
                original=f"https://linkedin.com/jobs/view/{i}",
                timestamp=f"2020{i:02d}01120000",
            )
            for i in range(1, 4)
        ]
        pages = [_make_fetched_page(r) for r in records]
        parsed = _make_parsed()

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.return_value = iter(records)
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_one.side_effect = pages
        mock_fetcher_cls.return_value = mock_fetcher

        with patch(
            "krm.phase0.collectors.linkedin_wayback.parse_modern_linkedin",
            return_value=parsed,
        ):
            result = collect_linkedin_wayback(2020)

        assert result == 3
        assert mock_upsert.call_count == 3
        mock_finish.assert_called_once_with(mock_conn, "linkedin-2020", 3, 3)


# ---------------------------------------------------------------------------
# Tests: Connection closed
# ---------------------------------------------------------------------------


class TestLinkedInConnectionClosed:
    """The database connection is always closed, even on error."""

    @patch("krm.phase0.collectors.linkedin_wayback.get_phase0_connection")
    @patch("krm.phase0.collectors.linkedin_wayback.WaybackFetcher")
    @patch("krm.phase0.collectors.linkedin_wayback.CdxQuerier")
    @patch("krm.phase0.collectors.linkedin_wayback.start_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.init_phase0_tables")
    @patch("krm.phase0.collectors.linkedin_wayback.finish_phase0_run")
    @patch("krm.phase0.collectors.linkedin_wayback.upsert_phase0_vacancy")
    def test_linkedin_connection_closed_on_exception(
        self,
        mock_upsert: MagicMock,
        mock_finish: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_querier_cls: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
    ) -> None:
        """Given an exception during processing,
        When collect_linkedin_wayback runs,
        Then conn.close() is still called via the finally block.
        """
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        mock_querier = MagicMock()
        mock_querier.query_paginated.side_effect = RuntimeError("CDX error")
        mock_querier_cls.return_value = mock_querier

        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher

        with pytest.raises(RuntimeError, match="CDX error"):
            collect_linkedin_wayback(2020)

        mock_conn.close.assert_called_once()


# ======================================================================
# Tests: hhru_wayback collector
# ======================================================================

_COLLECTOR_HHRU = "krm.phase0.collectors.hhru_wayback"


@pytest.fixture
def mock_hhru_config() -> MagicMock:
    """A mock Config with hhru_wayback defaults matching config.yaml."""
    cfg = MagicMock()
    cfg.phase0_hhru_wayback_years = [2010, 2025]
    cfg.phase0_cdx_endpoint = "https://web.archive.org/cdx/search/cdx"
    cfg.phase0_cdx_rate_limit_rps = 1
    return cfg


def _make_hhru_parsed() -> dict:
    """Return a minimal parsed hh.ru dict that survives map → normalise."""
    return {
        "id": "12345",
        "name": "Test Vacancy",
        "description": "A test job.",
        "employer": {"name": "Acme Corp"},
        "area": {"name": "Moscow"},
        "salary": {"from_": None, "to": None, "currency": None},
        "published_at": "2020-06-15",
        "experience": {"name": None},
        "key_skills": [],
        "professional_roles": [],
        "schedule": {"name": None},
        "_parser_version": "test_v1",
    }


class TestCollectHhruWayback:
    """Integration-style tests for collect_hhru_wayback()."""

    # -- Happy path -------------------------------------------------------

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record", side_effect=lambda d: d)
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback")
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_happy_path_stores_vacancies_and_calls_lifecycle(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """Given two CDX records (one legacy, one modern), both are
        parsed, mapped, normalised, and stored with the correct run_id."""
        legacy_rec = _make_cdx_record(
            original="http://hh.ru/vacancy.do?id=12345",
            timestamp="20150101120000",
        )
        modern_rec = _make_cdx_record(
            original="http://hh.ru/vacancy/67890",
            timestamp="20150615180000",
        )

        legacy_page = _make_fetched_page(legacy_rec)
        modern_page = _make_fetched_page(modern_rec)

        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([legacy_rec, modern_rec])

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch_one.side_effect = [legacy_page, modern_page]

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        parsed_legacy = _make_hhru_parsed()
        parsed_modern = _make_hhru_parsed()
        parsed_modern["id"] = "67890"
        parsed_modern["name"] = "Modern Vacancy"

        mock_parse_legacy.return_value = parsed_legacy
        mock_parse_modern.return_value = parsed_modern

        mock_map.side_effect = lambda parsed, url, ts: {
            **parsed,
            "_phase0_source": "wayback-hhru",
            "_phase0_capture_ts": ts,
            "_phase0_original_url": url,
        }

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        result = collect_hhru_wayback(2015, config=mock_hhru_config)

        assert result == 2
        mock_get_conn.assert_called_once()
        mock_init.assert_called_once_with(mock_conn)
        mock_start.assert_called_once_with(
            mock_conn, "wayback-hhru-2015", "wayback-hhru", "hh.ru/vacancy/*"
        )
        mock_finish.assert_called_once_with(mock_conn, "wayback-hhru-2015", 2, 2)
        mock_cdx.assert_called_once_with(
            endpoint=mock_hhru_config.phase0_cdx_endpoint,
            rate_limit_rps=mock_hhru_config.phase0_cdx_rate_limit_rps,
        )
        mock_querier.query_paginated.assert_called_once_with(
            "hh.ru/vacancy/*",
            from_ts="20150101",
            to_ts="20151231",
            status_filter=200,
        )
        mock_fetcher_cls.assert_called_once_with(
            rate_limit_rps=mock_hhru_config.phase0_cdx_rate_limit_rps
        )
        assert mock_fetcher.fetch_one.call_count == 2
        mock_parse_legacy.assert_called_once_with(
            legacy_page.html_content, legacy_rec.original
        )
        mock_parse_modern.assert_called_once_with(
            modern_page.html_content, modern_rec.original
        )
        assert mock_upsert.call_count == 2
        mock_upsert.assert_has_calls(
            [
                call(mock_conn, "wayback-hhru-2015", "12345", ANY),
                call(mock_conn, "wayback-hhru-2015", "67890", ANY),
            ]
        )

    # -- Parser selection -------------------------------------------------

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record", side_effect=lambda d: d)
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback", side_effect=lambda p, u, t: p)
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_selects_legacy_parser_for_do_urls(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """Given a URL containing '.do', the legacy parser is used."""
        rec = _make_cdx_record(
            original="http://hh.ru/vacancy.do?id=42",
            timestamp="20100101000000",
        )
        page = _make_fetched_page(rec)

        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([rec])

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch_one.return_value = page

        mock_get_conn.return_value = MagicMock()
        mock_parse_legacy.return_value = _make_hhru_parsed()

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        collect_hhru_wayback(2010, config=mock_hhru_config)

        mock_parse_legacy.assert_called_once()
        mock_parse_modern.assert_not_called()

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record", side_effect=lambda d: d)
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback", side_effect=lambda p, u, t: p)
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_selects_modern_parser_for_non_do_urls(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """Given a URL without '.do', the modern parser is used."""
        rec = _make_cdx_record(
            original="http://hh.ru/vacancy/99999",
            timestamp="20200615000000",
        )
        page = _make_fetched_page(rec)

        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([rec])

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch_one.return_value = page

        mock_get_conn.return_value = MagicMock()
        mock_parse_modern.return_value = _make_hhru_parsed()

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        collect_hhru_wayback(2020, config=mock_hhru_config)

        mock_parse_modern.assert_called_once()
        mock_parse_legacy.assert_not_called()

    # -- Parse failure is skipped -----------------------------------------

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record")
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback")
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_parse_failure_is_skipped_not_crashed(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """Given a vacancy that fails to parse, it is skipped and the
        collector continues processing the next one."""
        good_rec = _make_cdx_record(
            original="http://hh.ru/vacancy/111",
            timestamp="20200101000000",
        )
        bad_rec = _make_cdx_record(
            original="http://hh.ru/vacancy/222",
            timestamp="20200102000000",
        )

        good_page = _make_fetched_page(good_rec)
        bad_page = _make_fetched_page(bad_rec)

        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([good_rec, bad_rec])

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch_one.side_effect = [good_page, bad_page]

        mock_get_conn.return_value = MagicMock()

        mock_parse_modern.side_effect = [
            _make_hhru_parsed(),
            RuntimeError("Parser exploded"),
        ]
        mock_map.side_effect = lambda p, u, t: p
        mock_normalize.side_effect = lambda d: d

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        result = collect_hhru_wayback(2020, config=mock_hhru_config)

        assert result == 1
        mock_upsert.assert_called_once()
        mock_finish.assert_called_once_with(ANY, "wayback-hhru-2020", 2, 1)

    # -- Normalise failure is skipped --------------------------------------

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record", side_effect=ValueError("missing id"))
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback", side_effect=lambda p, u, t: p)
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_normalise_failure_is_skipped(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """Given a vacancy whose mapped dict fails normalise(), it is
        skipped and the collector continues."""
        rec = _make_cdx_record(
            original="http://hh.ru/vacancy/42",
            timestamp="20200101000000",
        )
        page = _make_fetched_page(rec)

        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([rec])

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch_one.return_value = page

        mock_get_conn.return_value = MagicMock()
        mock_parse_modern.return_value = _make_hhru_parsed()

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        result = collect_hhru_wayback(2020, config=mock_hhru_config)

        assert result == 0
        mock_upsert.assert_not_called()
        mock_finish.assert_called_once_with(ANY, "wayback-hhru-2020", 1, 0)

    # -- Empty CDX results ------------------------------------------------

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record")
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback")
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_empty_cdx_returns_zero(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """Given an empty CDX result set, the collector returns 0 and
        never calls the fetcher or parser."""
        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([])

        mock_get_conn.return_value = MagicMock()

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        result = collect_hhru_wayback(2018, config=mock_hhru_config)

        assert result == 0
        mock_fetcher_cls.return_value.fetch_one.assert_not_called()
        mock_parse_legacy.assert_not_called()
        mock_parse_modern.assert_not_called()
        mock_upsert.assert_not_called()
        mock_finish.assert_called_once_with(ANY, "wayback-hhru-2018", 0, 0)

    # -- Year validation --------------------------------------------------

    def test_year_too_low_raises_valueerror(
        self, mock_hhru_config: MagicMock
    ) -> None:
        """Given a year below the configured minimum, ValueError is raised."""
        mock_hhru_config.phase0_hhru_wayback_years = [2010, 2025]
        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        with pytest.raises(ValueError, match="outside configured range"):
            collect_hhru_wayback(2009, config=mock_hhru_config)

    def test_year_too_high_raises_valueerror(
        self, mock_hhru_config: MagicMock
    ) -> None:
        """Given a year above the configured maximum, ValueError is raised."""
        mock_hhru_config.phase0_hhru_wayback_years = [2010, 2025]
        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        with pytest.raises(ValueError, match="outside configured range"):
            collect_hhru_wayback(2026, config=mock_hhru_config)

    def test_year_at_lower_bound_passes(
        self, mock_hhru_config: MagicMock
    ) -> None:
        """Given a year equal to the minimum, it is accepted."""
        mock_hhru_config.phase0_hhru_wayback_years = [2010, 2025]
        with (
            patch(f"{_COLLECTOR_HHRU}.CdxQuerier") as mock_cdx,
            patch(f"{_COLLECTOR_HHRU}.WaybackFetcher"),
            patch(f"{_COLLECTOR_HHRU}.get_phase0_connection"),
            patch(f"{_COLLECTOR_HHRU}.init_phase0_tables"),
            patch(f"{_COLLECTOR_HHRU}.start_phase0_run"),
            patch(f"{_COLLECTOR_HHRU}.finish_phase0_run"),
        ):
            mock_cdx.return_value.query_paginated.return_value = iter([])

            from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

            result = collect_hhru_wayback(2010, config=mock_hhru_config)
            assert result == 0

    def test_year_at_upper_bound_passes(
        self, mock_hhru_config: MagicMock
    ) -> None:
        """Given a year equal to the maximum, it is accepted."""
        mock_hhru_config.phase0_hhru_wayback_years = [2010, 2025]
        with (
            patch(f"{_COLLECTOR_HHRU}.CdxQuerier") as mock_cdx,
            patch(f"{_COLLECTOR_HHRU}.WaybackFetcher"),
            patch(f"{_COLLECTOR_HHRU}.get_phase0_connection"),
            patch(f"{_COLLECTOR_HHRU}.init_phase0_tables"),
            patch(f"{_COLLECTOR_HHRU}.start_phase0_run"),
            patch(f"{_COLLECTOR_HHRU}.finish_phase0_run"),
        ):
            mock_cdx.return_value.query_paginated.return_value = iter([])

            from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

            result = collect_hhru_wayback(2025, config=mock_hhru_config)
            assert result == 0

    # -- Fetch failure is skipped -----------------------------------------

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record", side_effect=lambda d: d)
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback", side_effect=lambda p, u, t: p)
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_fetch_failure_is_skipped(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """Given a record that fails to fetch (returns None), it is
        skipped and continues with the next record."""
        good_rec = _make_cdx_record(
            original="http://hh.ru/vacancy/111",
            timestamp="20200101000000",
        )
        bad_rec = _make_cdx_record(
            original="http://hh.ru/vacancy/222",
            timestamp="20200102000000",
        )

        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([bad_rec, good_rec])

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch_one.side_effect = [None, _make_fetched_page(good_rec)]

        mock_get_conn.return_value = MagicMock()
        mock_parse_modern.return_value = _make_hhru_parsed()

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        result = collect_hhru_wayback(2020, config=mock_hhru_config)

        assert result == 1
        assert mock_fetcher.fetch_one.call_count == 2
        assert mock_parse_modern.call_count == 1
        mock_finish.assert_called_once_with(ANY, "wayback-hhru-2020", 1, 1)

    # -- run_id format ----------------------------------------------------

    @patch(f"{_COLLECTOR_HHRU}.upsert_phase0_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.normalize_record", side_effect=lambda d: d)
    @patch(f"{_COLLECTOR_HHRU}.map_hhru_wayback", side_effect=lambda p, u, t: p)
    @patch(f"{_COLLECTOR_HHRU}.parse_modern_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.parse_legacy_vacancy")
    @patch(f"{_COLLECTOR_HHRU}.finish_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.start_phase0_run")
    @patch(f"{_COLLECTOR_HHRU}.init_phase0_tables")
    @patch(f"{_COLLECTOR_HHRU}.get_phase0_connection")
    @patch(f"{_COLLECTOR_HHRU}.WaybackFetcher")
    @patch(f"{_COLLECTOR_HHRU}.CdxQuerier")
    def test_run_id_format_matches_wayback_hhru_year(
        self,
        mock_cdx: MagicMock,
        mock_fetcher_cls: MagicMock,
        mock_get_conn: MagicMock,
        mock_init: MagicMock,
        mock_start: MagicMock,
        mock_finish: MagicMock,
        mock_parse_legacy: MagicMock,
        mock_parse_modern: MagicMock,
        mock_map: MagicMock,
        mock_normalize: MagicMock,
        mock_upsert: MagicMock,
        mock_hhru_config: MagicMock,
    ) -> None:
        """The run_id is 'wayback-hhru-{year}'."""
        rec = _make_cdx_record(
            original="http://hh.ru/vacancy/42",
            timestamp="20230101000000",
        )
        page = _make_fetched_page(rec)

        mock_querier = mock_cdx.return_value
        mock_querier.query_paginated.return_value = iter([rec])

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch_one.return_value = page

        mock_get_conn.return_value = MagicMock()
        mock_parse_modern.return_value = _make_hhru_parsed()

        from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

        collect_hhru_wayback(2023, config=mock_hhru_config)

        expected_run_id = "wayback-hhru-2023"
        mock_start.assert_called_once_with(
            ANY, expected_run_id, "wayback-hhru", "hh.ru/vacancy/*"
        )
        mock_upsert.assert_called_once_with(
            ANY, expected_run_id, "12345", ANY
        )
        mock_finish.assert_called_once_with(ANY, expected_run_id, 1, 1)
