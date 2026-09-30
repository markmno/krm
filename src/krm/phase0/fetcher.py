"""Wayback Machine content fetcher with rate limiting and retry/backoff.

Provides :class:`FetchedPage` for capturing fetched HTML content and
:class:`WaybackFetcher` for downloading raw Wayback Machine snapshots
with configurable rate limiting, timeout, and exponential backoff retry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from krm.phase0.cdx import CdxRecord

logger = logging.getLogger(__name__)

#: Base URL for raw Wayback Machine captures (``id_`` suffix strips toolbar).
WAYBACK_BASE = "https://web.archive.org/web"


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """A single fetched Wayback Machine snapshot.

    Attributes:
        cdx_record:  The CDX index record associated with this capture.
        html_content:  The raw HTML body of the fetched page.
        content_type:  The ``Content-Type`` header value from the response.
        fetched_at:  ISO 8601 UTC timestamp of when this page was fetched.
    """

    cdx_record: CdxRecord
    html_content: str
    content_type: str
    fetched_at: str  # ISO 8601


class WaybackFetcher:
    """Synchronous Wayback Machine content fetcher.

    Downloads raw capture content via ``/web/{timestamp}id_/{url}`` with
    rate limiting and exponential-backoff retry for transient failures.

    Args:
        rate_limit_rps:  Maximum requests per second. Defaults to ``1``.
        timeout:  Request timeout in seconds. Defaults to ``30``.
        max_retries:  Max retry attempts for 5xx / timeouts. Defaults to ``3``.
        user_agent:  ``User-Agent`` header sent with every request.
    """

    def __init__(
        self,
        rate_limit_rps: int = 1,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: str = "KRM-Pipeline/0.2.0 (historical-job-market-dataset)",
    ) -> None:
        self._rate_limit_rps: int = rate_limit_rps
        self._min_interval: float = 1.0 / rate_limit_rps
        self._last_request_time: float = 0.0
        self._timeout: int = timeout
        self._max_retries: int = max_retries
        self._user_agent: str = user_agent
        self._client: httpx.Client = httpx.Client(
            timeout=timeout,  # type: ignore[call-arg]
            headers={"User-Agent": user_agent},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_one(self, timestamp: str, url: str) -> FetchedPage | None:
        """Fetch a single Wayback Machine snapshot.

        Builds the URL ``https://web.archive.org/web/{timestamp}id_/{url}``
        which returns the raw capture without the Wayback toolbar.

        Args:
            timestamp:  Wayback capture timestamp (``YYYYMMDDhhmmss``).
            url:  Original URL of the captured page.

        Returns:
            A :class:`FetchedPage` on success, or ``None`` if the page is
            unavailable (4xx) or all retries exhausted (5xx / timeout).
        """
        wayback_url = f"{WAYBACK_BASE}/{timestamp}id_/{url}"

        # Minimal CDX record for standalone fetches.
        cdx_record = CdxRecord(urlkey="", timestamp=timestamp, original=url)

        for attempt in range(self._max_retries + 1):
            self._rate_limit()
            try:
                response = self._client.get(wayback_url)
                self._last_request_time = time.monotonic()

                # 4xx – page not available.
                if 400 <= response.status_code < 500:
                    return None

                response.raise_for_status()  # raises HTTPStatusError on 5xx

                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("text/html"):
                    return None

                return FetchedPage(
                    cdx_record=cdx_record,
                    html_content=response.text,
                    content_type=content_type,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )

            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                self._last_request_time = time.monotonic()
                if attempt < self._max_retries:
                    backoff = 2**attempt  # 1, 2, 4 seconds
                    logger.debug(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        self._max_retries,
                        url,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "Failed to fetch %s after %d retries: %s",
                        url,
                        self._max_retries,
                        exc,
                    )
                    return None

        return None

    def fetch_batch(self, records: list[CdxRecord]) -> list[FetchedPage]:
        """Fetch pages for a batch of CDX records, respecting rate limits.

        Each *record* is fetched via :meth:`fetch_one` using the record's
        ``timestamp`` and ``original`` fields.  Failed fetches (returning
        ``None``) are silently dropped from the output.

        Args:
            records:  CDX index records to fetch.

        Returns:
            A list of successfully fetched :class:`FetchedPage` instances.
            May be shorter than *records* due to failures.
        """
        results: list[FetchedPage] = []
        for record in records:
            page = self.fetch_one(record.timestamp, record.original)
            if page is not None:
                results.append(page)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if needed to maintain the configured requests-per-second limit."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
