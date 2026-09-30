"""Trudvsem.ru open data API collector for Russian government labor vacancy data.

Collects vacancies from https://opendata.trudvsem.ru/api/v1, maps them into the
hh.ru API schema via schema.map_trudvsem(), and supports keyword text search
and region-code partitioning for large-scale collection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from krm.phase0.schema import map_trudvsem

# Trudvsem API limits: max results window is ~10,000 (offset cap).
# Use region_code and text partitioning for larger result sets.
MAX_OFFSET = 10_000
FIRST_PAGE_LIMIT = 99   # offset=0 supports any reasonable limit
NEXT_PAGE_LIMIT = 10    # text search caps absolute offset at ~110; region-only has no cap
TEXT_SEARCH_CAP = 110   # maximum absolute offset reachable with text search


@dataclass
class TrudvsemVacancy:
    """Raw vacancy fields from Trudvsem API response (actual API shape)."""

    id: str | None = None
    vacancy_id: str | None = None
    # Real API uses "job-name", not "vacancy_name"
    job_name: str | None = field(default=None, metadata={"api": "job-name"})
    vacancy_name: str | None = None  # kept for backward compat
    requirements: str | None = None
    duty: str | None = None
    company: dict[str, Any] | None = None
    company_name: str | None = None
    region: dict[str, Any] | None = None
    region_name: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    source: str | None = None
    creation_date: str | None = field(default=None, metadata={"api": "creation-date"})
    date_modify: str | None = None
    url: str | None = field(default=None, metadata={"api": "vac_url"})
    source_url: str | None = None
    skills: list[Any] | None = None
    category: dict[str, Any] | None = None


class TrudvsemCollector:
    """Collects vacancies from the Trudvsem.ru open data API.

    Collects ALL vacancies matched by text/region queries (no source filter),
    mapping each through schema.map_trudvsem() for normalisation into the
    hh.ru API shape.
    """

    _base_url: str
    _rate_limit_rps: int
    _last_request_time: float
    _client: httpx.Client

    def __init__(
        self,
        base_url: str = "https://opendata.trudvsem.ru/api/v1",
        rate_limit_rps: int = 1,
    ) -> None:
        self._base_url = base_url
        self._rate_limit_rps = rate_limit_rps
        self._last_request_time = 0.0
        self._client = httpx.Client(timeout=90.0)

    # -- rate limiting -----------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce minimum interval between API requests."""
        if self._rate_limit_rps <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        min_interval = 1.0 / self._rate_limit_rps
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    # -- item unwrapping ---------------------------------------------------------

    @staticmethod
    def _unwrap_item(item: dict[str, Any]) -> dict[str, Any]:
        """Unwrap the Trudvsem API response wrapper ``{"vacancy": {...}}``.

        The API wraps each vacancy inside a ``vacancy`` key. This method
        extracts the inner dict, falling back to the item itself if the
        key is absent (forward compatibility).
        """
        return item.get("vacancy", item)

    def _request_with_retry(
        self, params: dict[str, Any], max_retries: int = 5
    ) -> dict[str, Any]:
        """GET /vacancies with retry on transient server errors.

        Trudvsem returns HTTP 200 with ``{"status": "500"}`` on errors,
        so we check both the HTTP status and JSON status field.
        """
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                response = self._client.get(
                    f"{self._base_url}/vacancies",
                    params=params,
                )
                if response.status_code != 200:
                    if response.status_code == 429:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60.0)
                        continue
                    if 500 <= response.status_code < 600:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60.0)
                        continue
                    response.raise_for_status()

                data = response.json()
                json_status = str(data.get("status", "200"))
                if json_status != "200":
                    # Trudvsem returns HTTP 200 with JSON error status
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue

                return data
            except httpx.RequestError:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
        raise RuntimeError(f"Max retries ({max_retries}) exceeded for {params}")

    # -- single-page collection --------------------------------------------------

    def collect(
        self,
        date_from: str = "",
        date_to: str = "",
        offset: int = 0,
        limit: int = FIRST_PAGE_LIMIT,
        text: str | None = None,
        region_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch a single page of vacancies from Trudvsem."""
        self._rate_limit()
        capture_ts = datetime.now(timezone.utc).isoformat()
        params = self._build_params(offset, limit, date_from, date_to, text, region_code)
        data = self._request_with_retry(params)
        items: list[dict[str, Any]] = data["results"]["vacancies"]
        return [map_trudvsem(self._unwrap_item(v), capture_ts) for v in items]

    # -- parameter builder -------------------------------------------------------

    @staticmethod
    def _build_params(
        offset: int = 0,
        limit: int = FIRST_PAGE_LIMIT,
        date_from: str = "",
        date_to: str = "",
        text: str | None = None,
        region_code: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if date_from:
            params["modifiedFrom"] = date_from
        if date_to:
            params["modifiedTo"] = date_to
        if text:
            params["text"] = text
        if region_code:
            params["region_code"] = region_code
        return params

    # -- full collection with pagination -----------------------------------------

    def collect_all(
        self,
        date_from: str = "",
        date_to: str = "",
        text: str | None = None,
        region_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Paginate through all Trudvsem vacancies matching the query."""
        all_vacancies: list[dict[str, Any]] = []
        offset = 0
        capture_ts = datetime.now(timezone.utc).isoformat()

        self._rate_limit()
        params = self._build_params(offset, FIRST_PAGE_LIMIT, date_from, date_to, text, region_code)
        data = self._request_with_retry(params)
        total: int = data["meta"]["total"]
        items: list[dict[str, Any]] = data["results"]["vacancies"]

        mapped = [map_trudvsem(self._unwrap_item(v), capture_ts) for v in items]
        all_vacancies.extend(mapped)
        offset += FIRST_PAGE_LIMIT

        effective_limit = min(total, MAX_OFFSET)
        # Region-only browsing (no text) has no offset limitation — use full page size.
        # Text search caps absolute offset at TEXT_SEARCH_CAP (~110), so use small pages
        # and cap the total reachable results.
        if text:
            effective_limit = min(effective_limit, TEXT_SEARCH_CAP)
            page_limit = NEXT_PAGE_LIMIT
        else:
            page_limit = FIRST_PAGE_LIMIT

        while offset < effective_limit:
            self._rate_limit()
            params["offset"] = offset
            params["limit"] = page_limit
            page_data = self._request_with_retry(params)
            page_items: list[dict[str, Any]] = page_data["results"]["vacancies"]
            if not page_items:
                break
            all_vacancies.extend(
                [map_trudvsem(self._unwrap_item(v), capture_ts) for v in page_items]
            )
            offset += page_limit

        return all_vacancies

    # -- count (quick census) ----------------------------------------------------

    def count(
        self,
        text: str | None = None,
        region_code: str | None = None,
        date_from: str = "",
    ) -> int:
        """Return the total matching count without fetching all pages."""
        self._rate_limit()

        params: dict[str, Any] = {"limit": 1, "offset": 0}
        if text:
            params["text"] = text
        if region_code:
            params["region_code"] = region_code
        if date_from:
            params["modifiedFrom"] = date_from

        response = self._client.get(
            f"{self._base_url}/vacancies",
            params=params,
        )
        _ = response.raise_for_status()
        data = response.json()
        return int(data["meta"]["total"])

    # -- teardown ----------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
