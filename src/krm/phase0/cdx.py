"""Wayback Machine CDX API querier with pagination via resumeKey and rate limiting.

Provides :class:`CdxRecord` for parsed CDX index rows and :class:`CdxQuerier`
for querying the Wayback Machine CDX Server API.

Rate limiting enforces a minimum interval between requests to respect the
Internet Archive's rate-limit policy. Pagination uses the native ``resumeKey``
mechanism for memory-efficient traversal of large result sets.

.. warning::

    Combining ``matchType=prefix`` with a ``*`` wildcard in the URL pattern
    (e.g. ``url_pattern="example.com/*"`` or ``url_pattern="*.example.com"``)
    **silently returns an empty result set** because the CDX server interprets
    the ``*`` literally under prefix matching. Use ``matchType=domain`` with a
    bare domain instead.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

#: Wayback Machine CDX Server API endpoint.
CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"

#: Field list sent with every CDX query (``fl=…`` parameter).
_CDX_FIELDS = "timestamp,original,mimetype,statuscode,digest,length"


@dataclass(frozen=True, slots=True)
class CdxRecord:
    """A single row from a Wayback Machine CDX index query.

    Attributes:
        urlkey:  SURT-formatted URL key (e.g. ``"com,example)/"``).
        timestamp:  Capture timestamp in ``YYYYMMDDhhmmss`` format.
        original:  The full original URL as captured.
        mimetype:  MIME type of the captured resource, or ``None``.
        statuscode:  HTTP status code, or ``None``.
        digest:  SHA-1 digest of the captured content, or ``None``.
        length:  Content length in bytes, or ``None``.
    """

    urlkey: str
    timestamp: str
    original: str
    mimetype: str | None = None
    statuscode: str | None = None
    digest: str | None = None
    length: str | None = None

    @classmethod
    def from_row(cls, row: list[str]) -> CdxRecord:
        """Parse a CDX JSON response data row into a :class:`CdxRecord`.

        The ``fl=timestamp,original,mimetype,statuscode,digest,length`` query
        returns 6 fields (no ``urlkey``).  This method maps them by position:

        * ``urlkey`` — set to ``""`` (not in the default ``fl`` output)
        * ``timestamp`` — ``row[0]``
        * ``original`` — ``row[1]``
        * ``mimetype`` — ``row[2]``  (``None`` if empty)
        * ``statuscode`` — ``row[3]``  (``None`` if empty)
        * ``digest`` — ``row[4]``  (``None`` if empty)
        * ``length`` — ``row[5]``  (``None`` if empty)

        Args:
            row: A list of 6 string fields from the CDX ``fl`` output.

        Returns:
            A frozen :class:`CdxRecord` with parsed fields.
        """
        # Guard against short rows (defensive, not expected in normal use).
        padded = list(row)
        while len(padded) < 6:
            padded.append("")

        return cls(
            urlkey="",
            timestamp=padded[0] or "",
            original=padded[1] or "",
            mimetype=padded[2] or None,
            statuscode=padded[3] or None,
            digest=padded[4] or None,
            length=padded[5] or None,
        )


class CdxQuerier:
    """Synchronous CDX API querier with rate limiting and resumeKey pagination.

    Wraps a single ``httpx.Client`` instance. Rate limiting tracks the
    timestamp of the last request and sleeps to maintain the configured
    requests-per-second ceiling.

    Args:
        endpoint:  CDX API endpoint URL.  Defaults to :data:`CDX_ENDPOINT`.
        rate_limit_rps:  Maximum requests per second (default ``1``).
    """

    def __init__(
        self,
        endpoint: str = CDX_ENDPOINT,
        rate_limit_rps: int = 1,
    ) -> None:
        self._endpoint: str = endpoint
        self._rate_limit_rps: int = rate_limit_rps
        self._min_interval: float = 1.0 / rate_limit_rps
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        url_pattern: str,
        *,
        from_ts: str | None = None,
        to_ts: str | None = None,
        collapse: str | None = None,
        match_type: str | None = None,
        status_filter: int | None = 200,
        limit: int | None = None,
        extra_filters: list[str] | None = None,
    ) -> list[CdxRecord]:
        """Query the CDX API and return a single page of results.

        Args:
            url_pattern:  URL pattern to search for (e.g. ``"hh.ru/vacancy/*"``).
            from_ts:  Earliest capture timestamp (``YYYYMMDDhhmmss``).
            to_ts:  Latest capture timestamp (``YYYYMMDDhhmmss``).
            collapse:  Collapse results by field (e.g. ``"urlkey"``).
            match_type:  CDX match scope (``"exact"``, ``"prefix"``, ``"host"``,
                ``"domain"``).  Defaults to CDX server default (``"exact"``).
            status_filter:  HTTP status code to filter on.  Pass ``None`` to
                disable the ``filter=statuscode:…`` parameter.  Defaults to
                ``200``.
            limit:  Maximum number of results to return.
            extra_filters:  Additional CDX filter strings (e.g.
                ``["mimetype:text/html"]``).

        Returns:
            A list of :class:`CdxRecord` instances.
        """
        params: dict[str, str | int] = {
            "url": url_pattern,
            "output": "json",
            "fl": _CDX_FIELDS,
        }
        if match_type is not None:
            params["matchType"] = match_type
        if from_ts is not None:
            params["from"] = from_ts
        if to_ts is not None:
            params["to"] = to_ts
        if collapse is not None:
            params["collapse"] = collapse
        if status_filter is not None:
            params["filter"] = [f"statuscode:{status_filter}"]
        if extra_filters:
            filters = params.get("filter", [])
            if isinstance(filters, str):
                filters = [filters]
            filters.extend(extra_filters)
            params["filter"] = filters
        if limit is not None:
            params["limit"] = str(limit)

        rows = self._fetch_json_rows(params)
        # Skip the header row (first element is field names).
        data_rows = rows[1:] if rows else []
        return [CdxRecord.from_row(row) for row in data_rows if row]

    def query_paginated(
        self,
        url_pattern: str,
        **kwargs: object,
    ) -> Iterator[CdxRecord]:
        """Query the CDX API with ``resumeKey`` pagination, collecting all results.

        Uses the CDX ``resumeKey`` mechanism: the first request includes
        ``showResumeKey=true``; the last row of the response is either the
        resume key list or an empty list signalling end-of-pages.  Subsequent
        requests carry the resume key until exhaustion.

        Accepts the same keyword arguments as :meth:`query` (except
        ``limit``, which is ignored for paginated queries).

        Yields:
            One :class:`CdxRecord` at a time, across all pages.

        .. warning::

            Combining ``matchType=prefix`` with a ``*`` wildcard silently
            returns an empty result set.  Use ``matchType=domain`` with a
            bare domain instead.
        """
        # Extract known keyword args for building params.
        from_ts: str | None = kwargs.get("from_ts", None)  # pyright: ignore[reportAssignmentType]
        to_ts: str | None = kwargs.get("to_ts", None)  # pyright: ignore[reportAssignmentType]
        collapse: str | None = kwargs.get("collapse", None)  # pyright: ignore[reportAssignmentType]
        match_type: str | None = kwargs.get("match_type", None)  # pyright: ignore[reportAssignmentType]
        status_filter: int | None = kwargs.get("status_filter", 200)  # pyright: ignore[reportAssignmentType]

        params: dict[str, str | int] = {
            "url": url_pattern,
            "output": "json",
            "fl": _CDX_FIELDS,
            "showResumeKey": "true",
        }
        if match_type is not None:
            params["matchType"] = str(match_type)
        if from_ts is not None:
            params["from"] = str(from_ts)
        if to_ts is not None:
            params["to"] = str(to_ts)
        if collapse is not None:
            params["collapse"] = str(collapse)
        if status_filter is not None:
            params["filter"] = f"statuscode:{status_filter}"

        while True:
            rows = self._fetch_json_rows(params)

            if not rows:
                return  # empty response

            # The last "row" is the resume key: a list of strings (could be empty).
            resume_key = rows[-1]

            # All rows except the last are data rows (or the header row and data).
            data_rows = rows[:-1]

            # Yield records from data rows (skip header if present).
            start = 1 if data_rows and self._is_header(data_rows[0]) else 0
            for row in data_rows[start:]:
                yield CdxRecord.from_row(row)

            # If resume_key is an empty list, we're done.
            if not resume_key:
                return

            # Set the resumeKey for the next page.
            params["resumeKey"] = resume_key[0] if len(resume_key) == 1 else ",".join(resume_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if the last request was less than ``_min_interval`` seconds ago."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    @staticmethod
    def _is_header(row: list[str]) -> bool:
        """Return ``True`` if *row* looks like the CDX field-name header."""
        if not row:
            return False
        first = row[0].strip() if row[0] else ""
        return first in ("urlkey", "timestamp", "original")

    def _fetch_json_rows(self, params: dict[str, str | int]) -> list[list[str]]:
        """Issue a rate-limited GET, return parsed JSON list-of-lists.

        Returns:
            The JSON array from the CDX response (header + data rows).

        Raises:
            httpx.HTTPError:  On non-2xx responses.
        """
        self._rate_limit()

        with httpx.Client(timeout=120.0) as client:
            response = client.get(self._endpoint, params=params)
            self._last_request_time = time.monotonic()
            response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return []

        return data
