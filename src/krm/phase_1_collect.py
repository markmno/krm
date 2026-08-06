"""Phase 1: HH.ru API vacancy scraper with DuckDB storage.

Collects raw STEM vacancy data from the HeadHunter API for all configured
keywords, respecting rate limits with exponential backoff, and stores results
in DuckDB via the shared I/O layer.

Usage:
    from krm.config import Config
    from krm.phase_1_collect import collect

    config = Config("config.yaml")
    new_vacancies = collect(config)
    print(f"Collected {new_vacancies} new vacancies")
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from krm.config import Config
from krm.lib.io import get_connection, init_tables, upsert_raw_vacancy

HH_API_BASE = "https://api.hh.ru"
HH_USER_AGENT = "KRM-Pipeline/0.2.0 (STEM-labor-market-analysis; https://github.com/example/krm)"
HH_PER_PAGE = 100  # Maximum page size supported by HH.ru API
HH_MAX_RESULTS = 2000  # HH.ru API hard cap on search results
MAX_PAGES = HH_MAX_RESULTS // HH_PER_PAGE  # 20 pages


def collect(config: Config) -> int:
    """Search HH.ru API for all configured keywords and store raw vacancies.

    Args:
        config: Typed configuration with keywords, date range, role filters,
                and rate limiting parameters.

    Returns:
        Total number of vacancies collected across all keywords.
    """
    conn = get_connection()
    init_tables(conn)

    date_from_iso = f"{config.date_from}T00:00:00"
    total_collected = 0

    # Use professional_roles as the primary API filter.
    # categories and exclude_roles are stored as metadata for downstream phases.
    api_params: dict[str, Any] = {
        "date_from": date_from_iso,
        "per_page": HH_PER_PAGE,
        "order_by": "publication_time",
    }

    for keyword in config.keywords:
        run_id = _make_run_id(keyword)
        _insert_scrape_run(conn, run_id, keyword, config)

        collected = _collect_keyword(
            conn=conn,
            run_id=run_id,
            keyword=keyword,
            api_params=api_params,
            config=config,
        )

        _finalize_scrape_run(conn, run_id, collected)
        total_collected += collected
        print(f"  [{keyword}] Collected {collected} vacancies")

    conn.close()
    return total_collected


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_run_id(keyword: str) -> str:
    """Generate a unique run ID for a keyword scrape session."""
    safe_keyword = keyword.replace(" ", "_").lower()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{safe_keyword}_{ts}"


def _insert_scrape_run(conn: Any, run_id: str, keyword: str, config: Config) -> None:
    """Create a scrape_runs row at the start of a keyword collection."""
    query_params = {
        "date_from": config.date_from,
        "professional_roles": config.professional_roles,
        "categories": config.categories,
        "exclude_roles": config.exclude_roles,
        "rate_limit_rps": config.rate_limit_rps,
    }
    import json

    conn.execute(
        "INSERT INTO scrape_runs (run_id, keyword, query_params) VALUES (?, ?, ?)",
        [run_id, keyword, json.dumps(query_params, ensure_ascii=False)],
    )


def _finalize_scrape_run(conn: Any, run_id: str, count: int) -> None:
    """Update the scrape_run with completion time and fetched count."""
    conn.execute(
        "UPDATE scrape_runs SET completed_at = CURRENT_TIMESTAMP, vacancies_fetched = ? WHERE run_id = ?",
        [count, run_id],
    )


def _collect_keyword(
    conn: Any,
    run_id: str,
    keyword: str,
    api_params: dict[str, Any],
    config: Config,
) -> int:
    """Paginate through HH.ru API for one keyword and upsert all vacancies.

    Returns the number of vacancies stored for this keyword.
    """
    params = dict(api_params)
    params["text"] = keyword

    # Attach professional role filters from config.
    if config.professional_roles:
        params["professional_role"] = config.professional_roles

    # Categories and exclude_roles are stored in scrape_run metadata
    # for downstream phases. Professional roles are the primary API filter.

    collected = 0
    backoff = 1.0  # Initial backoff in seconds

    with httpx.Client(
        base_url=HH_API_BASE,
        headers={"User-Agent": HH_USER_AGENT},
        timeout=30.0,
    ) as client:
        for page in range(MAX_PAGES):
            params["page"] = page

            response = _request_with_backoff(client, params, backoff, config)
            if response is None:
                break

            data = response.json()
            items = data.get("items", [])

            if not items:
                break

            for item in items:
                vacancy_id = str(item["id"])
                upsert_raw_vacancy(conn, run_id, vacancy_id, item)
                collected += 1

            print(
                f"  [{keyword}] Page {page + 1}: fetched {len(items)} "
                f"(total found: {data.get('found', '?')})"
            )

            # Stop if we've reached the last page.
            if page + 1 >= data.get("pages", 0):
                break

    return collected


def _request_with_backoff(
    client: httpx.Client,
    params: dict[str, Any],
    backoff: float,
    config: Config,
) -> httpx.Response | None:
    """Issue a GET /vacancies request with rate limiting and exponential backoff.

    Args:
        client: httpx client with base_url already set.
        params: Query parameters for the /vacancies endpoint.
        backoff: Current backoff duration in seconds (mutated on retry).
        config: Pipeline configuration for rate_limit_rps.

    Returns:
        Response object on success, or None if we should stop paging.
    """
    min_interval = 1.0 / config.rate_limit_rps
    max_retries = 8

    for attempt in range(max_retries):
        time.sleep(min_interval)

        try:
            response = client.get("/vacancies", params=params)
        except httpx.RequestError as exc:
            print(f"  Request error: {exc}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            print(f"  Rate limited (429). Backing off {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue

        if response.status_code == 400:
            # HH.ru returns 400 when page exceeds available results.
            print(f"  Page out of range (400). Stopping pagination.")
            return None

        if response.status_code == 404:
            print(f"  Endpoint not found (404): {response.text[:200]}")
            return None

        if 500 <= response.status_code < 600:
            print(
                f"  Server error {response.status_code}. "
                f"Retrying in {backoff:.1f}s..."
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue

        # Unexpected status code.
        print(f"  Unexpected status {response.status_code}: {response.text[:200]}")
        response.raise_for_status()

    print(f"  Max retries ({max_retries}) exceeded. Stopping pagination.")
    return None
