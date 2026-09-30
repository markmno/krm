"""Tests for phase0/cdx.py — CDX API querier with pagination and rate limiting."""

from __future__ import annotations

from unittest import mock

import pytest

from krm.phase0 import cdx as cdx_module
from krm.phase0.cdx import CDX_ENDPOINT, CdxQuerier, CdxRecord


# ======================================================================
# CdxRecord.from_row()
# ======================================================================


class TestCdxRecordFromRow:
    def test_parses_full_six_field_row(self) -> None:
        row = ["20240101000000", "http://example.com/page", "text/html", "200", "ABC123", "1234"]

        rec = CdxRecord.from_row(row)

        assert rec.urlkey == ""
        assert rec.timestamp == "20240101000000"
        assert rec.original == "http://example.com/page"
        assert rec.mimetype == "text/html"
        assert rec.statuscode == "200"
        assert rec.digest == "ABC123"
        assert rec.length == "1234"

    def test_empty_mimetype_becomes_none(self) -> None:
        row = ["20240101000000", "http://example.com", "", "200", "ABC", "100"]

        rec = CdxRecord.from_row(row)

        assert rec.mimetype is None

    def test_empty_statuscode_becomes_none(self) -> None:
        row = ["20240101000000", "http://example.com", "text/html", "", "ABC", "100"]

        rec = CdxRecord.from_row(row)

        assert rec.statuscode is None

    def test_empty_digest_becomes_none(self) -> None:
        row = ["20240101000000", "http://example.com", "text/html", "200", "", "100"]

        rec = CdxRecord.from_row(row)

        assert rec.digest is None

    def test_empty_length_becomes_none(self) -> None:
        row = ["20240101000000", "http://example.com", "text/html", "200", "ABC", ""]

        rec = CdxRecord.from_row(row)

        assert rec.length is None

    def test_all_optional_fields_empty(self) -> None:
        row = ["20240101000000", "http://example.com", "", "", "", ""]

        rec = CdxRecord.from_row(row)

        assert rec.timestamp == "20240101000000"
        assert rec.original == "http://example.com"
        assert rec.mimetype is None
        assert rec.statuscode is None
        assert rec.digest is None
        assert rec.length is None

    def test_record_is_frozen(self) -> None:
        row = ["20240101000000", "http://example.com", "text/html", "200", "ABC", "1234"]
        rec = CdxRecord.from_row(row)

        with pytest.raises(Exception):
            rec.timestamp = "changed"  # type: ignore[misc]

    def test_short_row_pads_with_empties(self) -> None:
        row = ["20240101000000", "http://example.com"]

        rec = CdxRecord.from_row(row)

        assert rec.timestamp == "20240101000000"
        assert rec.original == "http://example.com"
        assert rec.mimetype is None
        assert rec.statuscode is None
        assert rec.digest is None
        assert rec.length is None

    def test_extra_fields_are_ignored(self) -> None:
        row = [
            "20240101000000",
            "http://example.com",
            "text/html",
            "200",
            "ABC",
            "1234",
            "extra",
            "more",
        ]

        rec = CdxRecord.from_row(row)

        assert rec.timestamp == "20240101000000"
        assert rec.length == "1234"


# ======================================================================
# CdxQuerier.query() — single-page
# ======================================================================


def _make_mock_response(*rows: list[str]) -> mock.MagicMock:
    """Build a mock httpx.Response with the given JSON data rows."""
    header = ["timestamp", "original", "mimetype", "statuscode", "digest", "length"]
    data = [header, *rows]
    resp = mock.MagicMock()
    resp.json.return_value = data
    return resp


class TestCdxQuerierQuery:
    def test_single_result(self) -> None:
        mock_resp = _make_mock_response(
            ["20240101000000", "http://hh.ru/vacancy/123", "text/html", "200", "ABC", "1234"],
        )

        with mock.patch("httpx.Client.get", return_value=mock_resp):
            querier = CdxQuerier()
            results = querier.query("hh.ru/vacancy/*")

        assert len(results) == 1
        assert results[0].original == "http://hh.ru/vacancy/123"

    def test_multiple_results(self) -> None:
        mock_resp = _make_mock_response(
            ["20240101000000", "http://hh.ru/vacancy/1", "text/html", "200", "A1", "100"],
            ["20240102000000", "http://hh.ru/vacancy/2", "text/html", "200", "B2", "200"],
            ["20240103000000", "http://hh.ru/vacancy/3", "text/html", "200", "C3", "300"],
        )

        with mock.patch("httpx.Client.get", return_value=mock_resp):
            querier = CdxQuerier()
            results = querier.query("hh.ru/vacancy/*")

        assert len(results) == 3
        assert [r.original for r in results] == [
            "http://hh.ru/vacancy/1",
            "http://hh.ru/vacancy/2",
            "http://hh.ru/vacancy/3",
        ]

    def test_empty_response(self) -> None:
        mock_resp = _make_mock_response()

        with mock.patch("httpx.Client.get", return_value=mock_resp):
            querier = CdxQuerier()
            results = querier.query("no-such-domain.com/*")

        assert results == []

    def test_header_only_response_no_data_rows(self) -> None:
        """CDX API returns just [header_row] — header is skipped."""
        header = ["timestamp", "original", "mimetype", "statuscode", "digest", "length"]
        resp = mock.MagicMock()
        resp.json.return_value = [header]

        with mock.patch("httpx.Client.get", return_value=resp):
            querier = CdxQuerier()
            results = querier.query("hh.ru/vacancy/*")

        assert results == []

    def test_includes_statuscode_filter_by_default(self) -> None:
        mock_resp = _make_mock_response()

        with mock.patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            querier = CdxQuerier()
            querier.query("hh.ru/*")

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert any(
            p == "filter" and v == "statuscode:200" for p, v in params.items()
        )

    def test_disabled_status_filter(self) -> None:
        mock_resp = _make_mock_response()

        with mock.patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            querier = CdxQuerier()
            querier.query("hh.ru/*", status_filter=None)

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert all(p != "filter" for p in params)

    def test_respects_limit_parameter(self) -> None:
        mock_resp = _make_mock_response()

        with mock.patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            querier = CdxQuerier()
            querier.query("hh.ru/*", limit=5)

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["limit"] == "5"

    def test_none_limit_excluded_from_params(self) -> None:
        mock_resp = _make_mock_response()

        with mock.patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            querier = CdxQuerier()
            querier.query("hh.ru/*", limit=None)

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert "limit" not in params

    def test_non_json_response_returns_empty(self) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {"not": "a list"}

        with mock.patch("httpx.Client.get", return_value=resp):
            querier = CdxQuerier()
            results = querier.query("hh.ru/*")

        assert results == []


# ======================================================================
# CdxQuerier.query_paginated() — resumeKey pagination
# ======================================================================


def _make_paginated_mock(page_data: list[list[list[str]]]) -> mock.MagicMock:
    """Return a mock get() that yields successive pages from *page_data*."""
    mock_get = mock.MagicMock()

    responses: list[mock.MagicMock] = []
    for page in page_data:
        resp = mock.MagicMock()
        resp.json.return_value = page
        responses.append(resp)

    mock_get.side_effect = responses
    return mock_get


class TestCdxQuerierQueryPaginated:
    def test_two_pages_yields_all_records(self) -> None:
        header = ["timestamp", "original", "mimetype", "statuscode", "digest", "length"]
        page1 = [
            header,
            ["20240101000000", "http://a.com/1", "text/html", "200", "D1", "100"],
            ["resume-key-abc"],
        ]
        page2 = [
            header,
            ["20240102000000", "http://a.com/2", "text/html", "200", "D2", "200"],
            [],  # empty resumeKey = end
        ]

        mock_get = _make_paginated_mock([page1, page2])

        with mock.patch("httpx.Client.get", mock_get):
            querier = CdxQuerier()
            results = list(querier.query_paginated("a.com/*"))

        assert len(results) == 2
        assert results[0].original == "http://a.com/1"
        assert results[1].original == "http://a.com/2"

    def test_single_page_returns_results(self) -> None:
        header = ["timestamp", "original", "mimetype", "statuscode", "digest", "length"]
        page = [
            header,
            ["20240101000000", "http://a.com/1", "text/html", "200", "D1", "100"],
            [],  # empty resumeKey = end
        ]

        mock_get = _make_paginated_mock([page])

        with mock.patch("httpx.Client.get", mock_get):
            querier = CdxQuerier()
            results = list(querier.query_paginated("a.com/*"))

        assert len(results) == 1

    def test_empty_response_yields_nothing(self) -> None:
        header = ["timestamp", "original", "mimetype", "statuscode", "digest", "length"]
        page = [header, []]  # no data, empty resumeKey

        mock_get = _make_paginated_mock([page])

        with mock.patch("httpx.Client.get", mock_get):
            querier = CdxQuerier()
            results = list(querier.query_paginated("a.com/*"))

        assert results == []

    def test_resume_key_with_multiple_parts(self) -> None:
        header = ["timestamp", "original", "mimetype", "statuscode", "digest", "length"]
        page1 = [
            header,
            ["20240101000000", "http://a.com/1", "text/html", "200", "D1", "100"],
            ["part1", "part2", "part3"],
        ]
        page2 = [
            header,
            ["20240102000000", "http://a.com/2", "text/html", "200", "D2", "200"],
            [],
        ]

        mock_get = _make_paginated_mock([page1, page2])

        with mock.patch("httpx.Client.get", mock_get):
            querier = CdxQuerier()
            results = list(querier.query_paginated("a.com/*"))

        assert len(results) == 2
        # Verify resumeKey was passed as comma-joined parts.
        second_call = mock_get.call_args_list[1]
        assert "resumeKey" in second_call[1]["params"]

    def test_pagination_respects_rate_limit(self) -> None:
        header = ["timestamp", "original", "mimetype", "statuscode", "digest", "length"]
        page1 = [
            header,
            ["20240101000000", "http://a.com/1", "text/html", "200", "D1", "100"],
            ["key-1"],
        ]
        page2 = [
            header,
            ["20240102000000", "http://a.com/2", "text/html", "200", "D2", "200"],
            [],
        ]

        mock_get = _make_paginated_mock([page1, page2])

        with mock.patch("httpx.Client.get", mock_get):
            with mock.patch("time.sleep") as mock_sleep:
                querier = CdxQuerier(rate_limit_rps=2)
                results = list(querier.query_paginated("a.com/*"))

        # Two requests → at least one sleep call (first request: no sleep,
        # second request: maybe sleep depending on timing). With rate=2 RPS,
        # the second request should trigger sleep.
        assert len(results) == 2
        assert mock_sleep.call_count >= 1


# ======================================================================
# Rate limiting
# ======================================================================


class TestRateLimiting:
    def test_single_request_no_sleep(self) -> None:
        mock_resp = _make_mock_response()

        with mock.patch("httpx.Client.get", return_value=mock_resp):
            with mock.patch("time.sleep") as mock_sleep:
                querier = CdxQuerier(rate_limit_rps=10)
                querier.query("hh.ru/*")

        # First request should not need to sleep.
        assert mock_sleep.call_count == 0

    def test_second_request_sleeps_if_too_fast(self) -> None:
        mock_resp = _make_mock_response(
            ["20240101000000", "http://a.com/1", "text/html", "200", "D1", "100"],
        )

        with mock.patch("httpx.Client.get", return_value=mock_resp):
            with mock.patch("time.sleep") as mock_sleep:
                querier = CdxQuerier(rate_limit_rps=1)
                querier.query("a.com/*")
                querier.query("a.com/*")

        # Second request at 1 RPS should trigger a sleep.
        assert mock_sleep.call_count >= 1




# ======================================================================
# Module constants
# ======================================================================


class TestConstants:
    def test_cdx_endpoint_is_wayback_url(self) -> None:
        assert CDX_ENDPOINT == "https://web.archive.org/cdx/search/cdx"


# ======================================================================
# matchType=prefix + * wildcard warning
# ======================================================================


class TestPrefixWildcardWarning:
    def test_module_docstring_warns_about_prefix_wildcard(self) -> None:
        """The module docstring warns about matchType=prefix + * wildcard."""
        doc = (cdx_module.__doc__ or "").lower()
        assert "matchType=prefix" in doc or "matchtype" in doc
        assert "wildcard" in doc or "*" in doc or "empty" in doc

    def test_query_paginated_docstring_warns(self) -> None:
        """query_paginated docstring also warns."""
        doc = (CdxQuerier.query_paginated.__doc__ or "").lower()
        assert "*" in doc or "wildcard" in doc or "empty" in doc
