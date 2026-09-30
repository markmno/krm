"""Phase 1: HH.ru API vacancy scraper with DuckDB storage (async).

Collects raw STEM vacancy data from the HeadHunter API for all configured
keywords, respecting rate limits with exponential backoff, and stores results
in DuckDB via the shared I/O layer. Uses ``httpx.AsyncClient`` so requests
share a connection pool and support an optional HTTP proxy (for when hh.ru
IP-blocks the host).

Usage:
    from krm.config import Config
    from krm.phase_1_collect import collect

    config = Config("config.yaml")
    new_vacancies = collect(config)
    print(f"Collected {new_vacancies} new vacancies")
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from krm.config import Config
from krm.lib.io import get_connection, init_tables, upsert_raw_vacancy

HH_API_BASE = "https://api.hh.ru"
HH_USER_AGENT = "KRM-Pipeline/0.2.0 (STEM-labor-market-analysis)"
HH_PER_PAGE = 100  # Maximum page size supported by HH.ru API
HH_MAX_RESULTS = 2000  # HH.ru API hard cap on search results
MAX_PAGES = HH_MAX_RESULTS // HH_PER_PAGE  # 20 pages

_MAX_RETRIES = 8


def collect(config: Config) -> int:
    """Search HH.ru API for all configured keywords and store raw vacancies.

    Args:
        config: Typed configuration with keywords, date range, role filters,
                rate limiting, and optional ``http_proxy``.

    Returns:
        Total number of vacancies collected across all keywords.
    """
    conn = get_connection()
    init_tables(conn)
    try:
        total = asyncio.run(_collect_async(config, conn))
    finally:
        conn.close()
    return total


async def _collect_async(config: Config, conn: Any) -> int:
    """Run keyword collection against a single shared async client."""
    api_params: dict[str, Any] = {
        "date_from": f"{config.date_from}T00:00:00",
        "per_page": HH_PER_PAGE,
        "order_by": "publication_time",
    }
    min_interval = 1.0 / config.rate_limit_rps
    proxy = config.http_proxy or None

    async with httpx.AsyncClient(
        base_url=HH_API_BASE,
        headers={"User-Agent": HH_USER_AGENT},
        timeout=30.0,
        proxy=proxy,
    ) as client:
        total = 0
        for keyword in config.keywords:
            run_id = _make_run_id(keyword)
            _insert_scrape_run(conn, run_id, keyword, config)
            collected = await _collect_keyword(
                client, conn, run_id, keyword, api_params, config, min_interval
            )
            _finalize_scrape_run(conn, run_id, collected)
            total += collected
            logger.info(f"  [{keyword}] Collected {collected} vacancies")
    return total


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


async def _collect_keyword(
    client: httpx.AsyncClient,
    conn: Any,
    run_id: str,
    keyword: str,
    api_params: dict[str, Any],
    config: Config,
    min_interval: float,
) -> int:
    """Paginate through HH.ru API for one keyword and upsert all vacancies."""
    params = dict(api_params)
    params["text"] = keyword

    # Professional roles are the primary API filter; categories and
    # exclude_roles are stored as metadata for downstream phases.
    if config.professional_roles:
        params["professional_role"] = config.professional_roles

    collected = 0
    for page in range(MAX_PAGES):
        params["page"] = page
        data = await _get_vacancies(client, params, min_interval)
        if data is None:
            break

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            upsert_raw_vacancy(conn, run_id, str(item["id"]), item)
            collected += 1

        logger.info(
            f"  [{keyword}] Page {page + 1}: fetched {len(items)} "
            f"(total found: {data.get('found', '?')})"
        )

        if page + 1 >= data.get("pages", 0):
            break

    return collected


async def _get_vacancies(
    client: httpx.AsyncClient,
    params: dict[str, Any],
    min_interval: float,
) -> dict[str, Any] | None:
    """GET /vacancies with rate limiting and exponential backoff.

    Returns:
        Parsed JSON dict on success, or ``None`` if we should stop paging.
    """
    backoff = 1.0  # seconds

    for _attempt in range(_MAX_RETRIES):
        await asyncio.sleep(min_interval)

        try:
            response = await client.get("/vacancies", params=params)
        except httpx.RequestError as exc:
            logger.error(f"  Request error: {exc}. Retrying in {backoff:.1f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            logger.error(f"  Rate limited (429). Backing off {backoff:.1f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue

        if response.status_code == 400:
            # HH.ru returns 400 when page exceeds available results.
            logger.info("Page out of range (400). Stopping pagination.")
            return None

        if response.status_code == 404:
            logger.error(f"  Endpoint not found (404): {response.text[:200]}")
            return None

        if 500 <= response.status_code < 600:
            logger.error(f"  Server error {response.status_code}. Retrying in {backoff:.1f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue

        # Unexpected status code.
        logger.warning(f"  Unexpected status {response.status_code}: {response.text[:200]}")
        response.raise_for_status()

    logger.info(f"  Max retries ({_MAX_RETRIES}) exceeded. Stopping pagination.")
    return None
