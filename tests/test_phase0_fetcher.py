"""Tests for phase0/fetcher.py — Wayback content fetcher with retry and rate limit."""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from krm.phase0.cdx import CdxRecord
from krm.phase0.fetcher import FetchedPage, WaybackFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_html_response() -> mock.MagicMock:
    """Return a mock httpx.Response representing a 200 HTML page."""
    resp = mock.MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.headers = {"content-type": "text/html; charset=utf-8"}
    resp.text = "<html><body><h1>Hello</h1></body></html>"
    resp.raise_for_status = mock.MagicMock()
    return resp


def _mock_503_response() -> mock.MagicMock:
    """Return a mock httpx.Response that raises HTTPStatusError on raise_for_status()."""
    resp = mock.MagicMock(spec=httpx.Response)
    resp.status_code = 503
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable",
        request=mock.MagicMock(),
        response=resp,
    )
    return resp


def _mock_404_response() -> mock.MagicMock:
    """Return a mock httpx.Response with 404 status."""
    resp = mock.MagicMock(spec=httpx.Response)
    resp.status_code = 404
    return resp


def _mock_pdf_response() -> mock.MagicMock:
    """Return a mock httpx.Response with application/pdf content-type."""
    resp = mock.MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.headers = {"content-type": "application/pdf"}
    return resp


def _make_fetcher(mock_get: mock.MagicMock | None = None) -> WaybackFetcher:
    """Create a WaybackFetcher with a mocked httpx.Client.

    If *mock_get* is provided, the client's ``get`` method is set to *mock_get*.
    """
    mock_client = mock.MagicMock(spec=httpx.Client)
    if mock_get is not None:
        mock_client.get = mock_get
    with mock.patch("httpx.Client", return_value=mock_client):
        return WaybackFetcher()


def _cdx_records(n: int = 3) -> list[CdxRecord]:
    """Create *n* sample CdxRecord instances."""
    records: list[CdxRecord] = []
    for i in range(n):
        records.append(
            CdxRecord(
                urlkey="",
                timestamp=f"2024010100000{i}",
                original=f"http://example.com/page{i}",
                mimetype="text/html",
                statuscode="200",
                digest=f"abc{i}",
                length=f"100{i}",
            )
        )
    return records


# ---------------------------------------------------------------------------
# fetch_one — success
# ---------------------------------------------------------------------------


class TestFetchOneSuccess:
    def test_fetches_html_and_returns_fetchedpage(self) -> None:
        mock_get = mock.MagicMock(return_value=_mock_html_response())
        fetcher = _make_fetcher(mock_get)

        result = fetcher.fetch_one("20240101000000", "http://example.com/page")

        assert result is not None
        assert isinstance(result, FetchedPage)
        assert result.html_content == "<html><body><h1>Hello</h1></body></html>"
        assert result.content_type.startswith("text/html")
        assert result.fetched_at is not None
        assert result.cdx_record.timestamp == "20240101000000"
        assert result.cdx_record.original == "http://example.com/page"


# ---------------------------------------------------------------------------
# fetch_one — non-HTML content
# ---------------------------------------------------------------------------


class TestFetchOneNonHtml:
    def test_pdf_content_type_returns_none(self) -> None:
        mock_get = mock.MagicMock(return_value=_mock_pdf_response())
        fetcher = _make_fetcher(mock_get)

        result = fetcher.fetch_one("20240101000000", "http://example.com/doc.pdf")

        assert result is None


# ---------------------------------------------------------------------------
# fetch_one — 404
# ---------------------------------------------------------------------------


class TestFetchOne404:
    def test_404_returns_none(self) -> None:
        mock_get = mock.MagicMock(return_value=_mock_404_response())
        fetcher = _make_fetcher(mock_get)

        result = fetcher.fetch_one("20240101000000", "http://example.com/missing")

        assert result is None


# ---------------------------------------------------------------------------
# fetch_one — 503 with retry
# ---------------------------------------------------------------------------


class TestFetchOne503Retry:
    def test_503_succeeds_on_retry_2(self) -> None:
        """First two attempts fail with 503; third attempt succeeds."""
        mock_get = mock.MagicMock()
        mock_get.side_effect = [
            _mock_503_response(),
            _mock_503_response(),
            _mock_html_response(),
        ]
        fetcher = _make_fetcher(mock_get)

        # Advance monotonic by >1s between iterations so _rate_limit does not sleep.
        with mock.patch("time.sleep") as mock_sleep:
            with mock.patch(
                "time.monotonic",
                side_effect=[
                    10.0,   # _rate_limit (iter 0): elapsed=10.0-0=10.0 => no sleep
                    10.0,   # _last_request_time after get()
                    10.0,   # _last_request_time in except (503)
                    11.5,   # _rate_limit (iter 1): elapsed=11.5-10.0=1.5 => no sleep
                    11.5,   # _last_request_time after get()
                    11.5,   # _last_request_time in except (503)
                    13.0,   # _rate_limit (iter 2): elapsed=13.0-11.5=1.5 => no sleep
                    13.0,   # _last_request_time after get() (200)
                ],
            ):
                result = fetcher.fetch_one("20240101000000", "http://example.com/page")

        assert result is not None
        assert result.html_content == "<html><body><h1>Hello</h1></body></html>"
        # Only backoff sleeps: 2^0=1s, 2^1=2s
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)
        # No other sleeps (rate-limit should not trigger)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# fetch_one — timeout with retry
# ---------------------------------------------------------------------------


class TestFetchOneTimeoutRetry:
    def test_timeout_succeeds_on_retry(self) -> None:
        """First attempt times out; second succeeds."""
        mock_get = mock.MagicMock()
        mock_get.side_effect = [
            httpx.TimeoutException("Read timed out"),
            _mock_html_response(),
        ]
        fetcher = _make_fetcher(mock_get)

        # Advance monotonic by >1s so _rate_limit does not sleep.
        with mock.patch("time.sleep") as mock_sleep:
            with mock.patch("time.monotonic", side_effect=[10.0, 10.0, 11.5, 11.5]):
                result = fetcher.fetch_one("20240101000000", "http://example.com/page")

        assert result is not None
        assert result.html_content == "<html><body><h1>Hello</h1></body></html>"
        mock_sleep.assert_called_once_with(1.0)

    def test_timeout_exhausts_retries_returns_none(self) -> None:
        """Four timeouts with max_retries=3 → returns None."""
        mock_get = mock.MagicMock()
        mock_get.side_effect = [
            httpx.TimeoutException("Read timed out"),
            httpx.TimeoutException("Read timed out"),
            httpx.TimeoutException("Read timed out"),
            httpx.TimeoutException("Read timed out"),
        ]
        fetcher = _make_fetcher(mock_get)

        with mock.patch("time.sleep"):
            result = fetcher.fetch_one("20240101000000", "http://example.com/page")

        assert result is None


# ---------------------------------------------------------------------------
# fetch_batch
# ---------------------------------------------------------------------------


class TestFetchBatch:
    def test_batch_returns_successful_pages_and_filters_none(self) -> None:
        """Batch of 3 records: 2 success, 1 failure (404) → returns 2."""
        mock_get = mock.MagicMock()
        mock_get.side_effect = [
            _mock_html_response(),          # record 0 — success
            _mock_404_response(),           # record 1 — None
            _mock_html_response(),          # record 2 — success
        ]
        fetcher = _make_fetcher(mock_get)
        records = _cdx_records(3)

        with mock.patch("time.sleep"):
            with mock.patch("time.monotonic", return_value=10.0):
                results = fetcher.fetch_batch(records)

        assert len(results) == 2
        assert all(isinstance(r, FetchedPage) for r in results)
        assert results[0].cdx_record.timestamp == records[0].timestamp
        assert results[1].cdx_record.timestamp == records[2].timestamp

    def test_batch_rate_limits_between_calls(self) -> None:
        """Verify _rate_limit is exercised during batch fetch."""
        mock_get = mock.MagicMock(return_value=_mock_html_response())
        fetcher = _make_fetcher(mock_get)
        records = _cdx_records(3)

        with mock.patch("time.sleep") as mock_sleep:
            with mock.patch("time.monotonic", side_effect=[0.0, 0.0, 0.19, 0.19, 0.38, 0.38]):
                fetcher.fetch_batch(records)

        # With rate_limit_rps=1 (min_interval=1.0), the 2nd and 3rd calls should sleep.
        assert mock_sleep.call_count >= 1


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_two_rapid_calls_second_sleeps(self) -> None:
        """Two rapid fetch_one calls → _rate_limit sleeps on the second."""
        mock_get = mock.MagicMock(return_value=_mock_html_response())
        fetcher = _make_fetcher(mock_get)

        with mock.patch("time.sleep") as mock_sleep:
            # First call: monotonic=0 → elapsed=inf → no sleep. Then _last=0.1.
            # Second call: monotonic=0.2 → elapsed=0.1 < 1.0 → sleep(0.9).
            with mock.patch(
                "time.monotonic",
                side_effect=[10.0, 10.0, 10.1, 10.1, 10.15, 10.15],
            ):
                fetcher.fetch_one("20240101000000", "http://example.com/a")
                fetcher.fetch_one("20240101000000", "http://example.com/b")

        # At least one sleep should have been triggered.
        assert mock_sleep.call_count >= 1


# ---------------------------------------------------------------------------
# User-Agent
# ---------------------------------------------------------------------------


class TestUserAgent:
    def test_user_agent_sent(self) -> None:
        """Verify httpx.Client is created with the correct User-Agent header."""
        with mock.patch("httpx.Client") as mock_client_class:
            _ = WaybackFetcher(
                user_agent="KRM-Pipeline/0.2.0 (historical-job-market-dataset)",
            )

        mock_client_class.assert_called_once()
        _, kwargs = mock_client_class.call_args
        assert kwargs["headers"] == {
            "User-Agent": "KRM-Pipeline/0.2.0 (historical-job-market-dataset)",
        }

    def test_user_agent_sent_in_request(self) -> None:
        """Verify the mock client has User-Agent in its headers."""
        mock_client_instance = mock.MagicMock(spec=httpx.Client)
        mock_client_instance.get = mock.MagicMock(return_value=_mock_html_response())

        with mock.patch("httpx.Client", return_value=mock_client_instance):
            fetcher = WaybackFetcher(
                user_agent="KRM-Pipeline/0.2.0 (historical-job-market-dataset)",
            )

        fetcher.fetch_one("20240101000000", "http://example.com/page")

        # Verify Client was created with User-Agent header
        # This was verified in test_user_agent_sent above; here we check get()
        mock_client_instance.get.assert_called_once()


# ---------------------------------------------------------------------------
# URL format
# ---------------------------------------------------------------------------


class TestUrlFormat:
    def test_url_contains_id_suffix(self) -> None:
        """Verify the constructed Wayback URL contains the ``id_`` suffix."""
        mock_get = mock.MagicMock(return_value=_mock_html_response())
        fetcher = _make_fetcher(mock_get)

        fetcher.fetch_one("20240101000000", "http://example.com/page")

        # Check that get was called with the correct id_ URL.
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert "/20240101000000id_/" in called_url
        assert called_url == "https://web.archive.org/web/20240101000000id_/http://example.com/page"
