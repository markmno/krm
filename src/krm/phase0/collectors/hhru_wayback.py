"""hh.ru Wayback Machine collector — orchestrates CDX → fetch → parse → map → store.

Collects historical hh.ru vacancy pages from the Wayback Machine using
daily CDX query windows (no collapse — client-side dedup) to avoid 504
timeouts caused by server-side urlkey aggregation.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from krm.phase0.cdx import CdxQuerier
from krm.phase0.fetcher import WaybackFetcher
from krm.phase0.parsers.hhru import parse_legacy_vacancy, parse_modern_vacancy
from krm.phase0.schema import enrich_record, map_hhru_wayback, normalize_record
from krm.phase0.storage import (
    finish_phase0_run,
    get_phase0_connection,
    init_phase0_tables,
    start_phase0_run,
    upsert_phase0_vacancy,
)

if TYPE_CHECKING:
    from krm.config import Config

logger = logging.getLogger(__name__)

_CDX_URL_PATTERN = "hh.ru/vacancy/*"


def collect_hhru_wayback(year: int, config: Config | None = None) -> int:
    """Collect hh.ru Wayback Machine vacancy captures for a single year.

    Uses daily CDX query windows WITHOUT ``collapse=urlkey`` to avoid
    504 timeouts.  Deduplication is done client-side by vacancy ID
    (extracted from the hh.ru URL).

    Args:
        year: Target year (e.g. ``2020``).
        config: Pipeline configuration.

    Returns:
        Number of vacancies successfully stored.
    """
    if config is None:
        from krm.config import Config
        config = Config()

    years = config.phase0_hhru_wayback_years
    min_year, max_year = min(years), max(years)
    if year < min_year or year > max_year:
        raise ValueError(
            f"Year {year} is outside configured range [{min_year}, {max_year}]"
        )

    run_id = f"wayback-hhru-{year}-{datetime.now().strftime('%m%d_%H%M')}"

    conn = get_phase0_connection()
    init_phase0_tables(conn)
    start_phase0_run(conn, run_id, "wayback-hhru", _CDX_URL_PATTERN)

    querier = CdxQuerier(
        endpoint=config.phase0_cdx_endpoint,
        rate_limit_rps=config.phase0_cdx_rate_limit_rps,
    )

    fetcher = WaybackFetcher(rate_limit_rps=config.phase0_cdx_rate_limit_rps)

    stored = 0
    fetched = 0
    skipped = 0
    seen_urls: set[str] = set()

    current_date = datetime(year, 1, 1)
    end_date = min(datetime(year + 1, 1, 1) - timedelta(days=1),
                   datetime.now())

    while current_date <= end_date:
        from_ts = current_date.strftime("%Y%m%d")
        to_date = min(current_date + timedelta(days=6), end_date)
        to_ts = to_date.strftime("%Y%m%d")

        try:
            records = querier.query(
                _CDX_URL_PATTERN,
                from_ts=from_ts,
                to_ts=to_ts,
                collapse="urlkey",
                status_filter=200,
                extra_filters=["mimetype:text/html"],
            )
        except Exception:
            logger.warning("CDX failed for %s–%s, skipping", from_ts, to_ts)
            current_date = to_date + timedelta(days=1)
            continue

        for record in records:
            url = record.original
            if url in seen_urls:
                continue
            seen_urls.add(url)

            page = fetcher.fetch_one(record.timestamp, url)
            if page is None:
                skipped += 1
                continue
            fetched += 1

            try:
                if ".do" in url:
                    parsed = parse_legacy_vacancy(page.html_content, url)
                else:
                    parsed = parse_modern_vacancy(page.html_content, url)

                mapped = map_hhru_wayback(dict(parsed), url, record.timestamp)
                normalized = normalize_record(mapped)
                enriched = enrich_record(normalized)
                upsert_phase0_vacancy(conn, run_id, enriched["id"], enriched)
                stored += 1
            except Exception:
                skipped += 1

        if (fetched + skipped) % 50 == 0:
            print(f"  {year}: {from_ts}–{to_ts} | fetched={fetched} stored={stored}")

        current_date = to_date + timedelta(days=1)

    finish_phase0_run(conn, run_id, fetched, stored)
    logger.info("Completed %s: fetched=%d stored=%d skipped=%d",
                run_id, fetched, stored, skipped)
    conn.close()
    return stored
