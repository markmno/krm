"""LinkedIn Wayback Machine collector via CDX API and Wayback Machine fetcher.

Provides :func:`collect_linkedin_wayback` for collecting historical LinkedIn
job postings from the Wayback Machine, with era-based URL pattern detection
and era-specific parsers.

Era mapping:
    - 2013–2016: ``linkedin.com/jobs2/view/*``  → parse_legacy_linkedin
    - 2019–2026: ``linkedin.com/jobs/view/*``   → parse_modern_linkedin
    - 2017–2018: skipped (no LinkedIn Wayback coverage)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from krm.phase0.cdx import CdxQuerier
from krm.phase0.fetcher import WaybackFetcher
from krm.phase0.parsers.linkedin import parse_legacy_linkedin, parse_modern_linkedin
from krm.phase0.schema import enrich_record, map_linkedin_wayback, normalize_record
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

# ---------------------------------------------------------------------------
# Era definitions
# ---------------------------------------------------------------------------

_LEGACY_ERA = range(2013, 2017)  # 2013–2016
_MODERN_ERA = range(2019, 2027)  # 2019–2026
_SKIPPED_ERA = (2017, 2018)

_LEGACY_URL_PATTERN = "linkedin.com/jobs2/view/*"
_MODERN_URL_PATTERN = "linkedin.com/jobs/view/*"


def _era_skip_message(year: int) -> str:
    """Return a human-readable skip message for a given year."""
    return f"No LinkedIn Wayback coverage in {year}"


def _url_pattern_for_year(year: int) -> str:
    """Return the CDX URL pattern for the given year.

    Raises:
        ValueError: If the year is outside all supported eras.
    """
    if year in _LEGACY_ERA:
        return _LEGACY_URL_PATTERN
    if year in _MODERN_ERA:
        return _MODERN_URL_PATTERN
    if year in _SKIPPED_ERA:
        raise ValueError(_era_skip_message(year))
    raise ValueError(f"Year {year} is outside supported LinkedIn range (2013–2026)")


# ---------------------------------------------------------------------------
# Main collector entry point
# ---------------------------------------------------------------------------


def collect_linkedin_wayback(year: int, config: Config | None = None) -> int:  # noqa: ARG001
    """Collect LinkedIn job postings from the Wayback Machine for a given year.

    Pipeline:
        1. Initialize storage and start a run (run_id = ``linkedin-{year}``).
        2. Determine the era-specific URL pattern and query CDX.
        3. Fetch each capture, parse with the correct era parser, map to the
           hh.ru schema, normalize, and store.
        4. Finish the run and return the count of stored records.

    Args:
        year: Target year (2013–2026 excl. 2017–2018).
        config: Optional :class:`~krm.config.Config` instance.  Currently
            unused; accepted for consistency with other collectors.

    Returns:
        Number of records stored.  Returns 0 for skipped years (2017–2018).

    Raises:
        ValueError: If *year* is outside the supported range (2013–2026).
    """
    run_id = f"linkedin-{year}"

    # ── 0. Skip 2017–2018 ─────────────────────────────────────────────────
    if year in _SKIPPED_ERA:
        msg = _era_skip_message(year)
        logger.info(msg)
        return 0

    # ── 1. Determine era-specific URL pattern ─────────────────────────────
    url_pattern = _url_pattern_for_year(year)

    # ── 2. Initialize storage + start run ──────────────────────────────────
    conn = get_phase0_connection()
    init_phase0_tables(conn)
    start_phase0_run(
        conn,
        run_id=run_id,
        source="linkedin_wayback",
        url_pattern=url_pattern,
        params={"year": year},
    )

    # ── 3. Create CDX querier + Wayback fetcher ────────────────────────────
    from_ts = f"{year}0101"
    to_ts = f"{year}1231"

    querier = CdxQuerier()
    fetcher = WaybackFetcher()

    # ── 4. Pipeline: CDX → fetch → parse → map → normalize → store ────────
    fetched: int = 0
    stored: int = 0

    try:
        for cdx_record in querier.query_paginated(
            url_pattern,
            from_ts=from_ts,
            to_ts=to_ts,
            status_filter=200,
            match_type="domain",
        ):
            fetched += 1

            # Fetch the snapshot
            page = fetcher.fetch_one(cdx_record.timestamp, cdx_record.original)
            if page is None:
                logger.debug(
                    "Skipping %s (fetch returned None)", cdx_record.original
                )
                continue

            # ── Era detection: choose the correct parser ─────────────────
            try:
                if "/jobs2/view/" in cdx_record.original:
                    parsed = parse_legacy_linkedin(
                        page.html_content, cdx_record.original
                    )
                else:
                    parsed = parse_modern_linkedin(
                        page.html_content, cdx_record.original
                    )
            except Exception:
                logger.warning(
                    "Parse failed for %s", cdx_record.original, exc_info=True
                )
                continue

            # ── Map to hh.ru schema + normalize ──────────────────────────
            mapped = map_linkedin_wayback(
                parsed, cdx_record.original, cdx_record.timestamp
            )

            try:
                normalized = normalize_record(mapped)
                enriched = enrich_record(normalized)
            except ValueError:
                logger.warning(
                    "Normalize failed for %s", cdx_record.original, exc_info=True
                )
                continue

            # ── Store ────────────────────────────────────────────────────
            upsert_phase0_vacancy(conn, run_id, enriched["id"], enriched)
            stored += 1

    finally:
        # ── 5. Finish run (always, even on exception) ─────────────────────
        finish_phase0_run(conn, run_id, fetched, stored)
        conn.close()

    logger.info(
        "LinkedIn %d: fetched=%d stored=%d",
        year,
        fetched,
        stored,
    )
    return stored
