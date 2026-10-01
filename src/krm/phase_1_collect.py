"""Phase 1 (collect): HH.ru constants and scrape-run helpers.

The live collection path is the website scraper ``krm.phase_1_site.collect_site``
(no API key). This module provides the scrape-run bookkeeping helpers that
``phase_1_site`` imports, plus the HH.ru API constants that describe the upstream
API contract.

Live surface:
    HH_API_BASE, HH_USER_AGENT, HH_PER_PAGE, HH_MAX_RESULTS, MAX_PAGES
    _make_run_id, _insert_scrape_run, _finalize_scrape_run
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from krm.config import Config

HH_API_BASE = "https://api.hh.ru"
HH_USER_AGENT = "KRM-Pipeline/0.2.0 (STEM-labor-market-analysis)"
HH_PER_PAGE = 100  # Maximum page size supported by HH.ru API
HH_MAX_RESULTS = 2000  # HH.ru API hard cap on search results
MAX_PAGES = HH_MAX_RESULTS // HH_PER_PAGE


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
