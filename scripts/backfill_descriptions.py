"""Backfill null descriptions + key_skills for a domain DuckDB.

During the bulk hh.ru scrape the detail pages were rate-limited, so ~99% of
vacancies were stored with ``description = null`` and empty ``key_skills``
(the scrape fell back to the search-list item, which carries only header
fields).  This script re-fetches the detail pages for those vacancies and
updates the stored JSON in place.

Usage:
    python -m scripts.backfill_descriptions physics
    python -m scripts.backfill_descriptions physics --concurrency 4 --delay 1.0
    python -m scripts.backfill_descriptions physics --limit 50 --dry-run

Flags:
    --concurrency N   concurrent detail fetches (default 4)
    --delay SEC       inter-request delay via a global rate limiter (default 0.8)
    --retries N       extra passes over still-null vacancies (default 2)
    --limit N         only process the first N null vacancies (for testing)
    --dry-run         fetch + parse but do not write to the DB
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import duckdb
import httpx
from loguru import logger

from krm.phase_1_site import _fetch_state, _strip_html

HH_SITE_HOST = "https://hh.ru"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
}


class RateLimiter:
    """Global async rate limiter bounding total requests-per-second."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
            self._next = time.monotonic() + self._interval


def _extract_key_skills(vv: dict) -> list[dict]:
    """Map ``vacancyView.keySkills`` onto the stored ``[{"name": ...}]`` shape."""
    raw = vv.get("keySkills") or {}
    if isinstance(raw, dict):
        items = raw.get("keySkill") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    out: list[dict] = []
    for s in items:
        if isinstance(s, str):
            name = s.strip()
        elif isinstance(s, dict):
            name = (s.get("name") or "").strip()
        else:
            continue
        if name:
            out.append({"name": name})
    return out


async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    limiter: RateLimiter,
    vacancy_id: str,
) -> tuple[str, str | None, list[dict] | None, str | None]:
    """Fetch one detail page and return (id, description, key_skills, error)."""
    async with sem:
        await limiter.wait()
        try:
            state = await _fetch_state(client, f"{HH_SITE_HOST}/vacancy/{vacancy_id}", {})
        except Exception as exc:  # noqa: BLE001 — scrape boundary
            return vacancy_id, None, None, f"fetch error: {exc}"

        vv = state.get("vacancyView")
        if not vv:
            return vacancy_id, None, None, "no vacancyView (rate-limited/captcha)"

        desc = _strip_html(vv.get("description"))
        if not desc:
            return vacancy_id, None, None, "null description (rate-limited)"

        ks = _extract_key_skills(vv)
        return vacancy_id, desc, ks, None


def _load_null_records(
    conn: duckdb.DuckDBPyConnection, limit: int | None
) -> dict[str, dict]:
    """Return ``{id: data_dict}`` for vacancies with null/empty description."""
    q = (
        "SELECT id, data FROM raw_vacancies "
        "WHERE data->>'description' IS NULL OR length(data->>'description') = 0 "
        "ORDER BY id"
    )
    rows = conn.execute(q).fetchall()
    records: dict[str, dict] = {}
    for vid, data_str in rows:
        try:
            records[vid] = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            records[vid] = {}
    if limit:
        keys = list(records.keys())[:limit]
        records = {k: records[k] for k in keys}
    return records


def _write_one(
    conn: duckdb.DuckDBPyConnection,
    records: dict[str, dict],
    vacancy_id: str,
    desc: str,
    key_skills: list[dict],
) -> None:
    """Update one vacancy's stored JSON with the recovered fields."""
    rec = records[vacancy_id]
    rec["description"] = desc
    rec["key_skills"] = key_skills
    conn.execute(
        "UPDATE raw_vacancies SET data = ? WHERE id = ?",
        [json.dumps(rec, ensure_ascii=False), vacancy_id],
    )


async def _run_pass(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    limiter: RateLimiter,
    conn: duckdb.DuckDBPyConnection,
    records: dict[str, dict],
    dry_run: bool,
) -> tuple[int, dict[str, int]]:
    """Fetch all ``records`` keys, updating the DB as results stream in."""
    ids = list(records.keys())
    filled = 0
    errors: dict[str, int] = {}
    pending = [_fetch_one(client, sem, limiter, vid) for vid in ids]
    done = 0
    for fut in asyncio.as_completed(pending):
        vacancy_id, desc, ks, err = await fut
        done += 1
        if err is not None:
            errors[err] = errors.get(err, 0) + 1
        else:
            assert desc is not None and ks is not None
            if not dry_run:
                _write_one(conn, records, vacancy_id, desc, ks)
            filled += 1
        if done % 200 == 0:
            logger.info(f"  progress: {done}/{len(ids)} (filled {filled})")
    return filled, errors


async def _backfill(
    domain: str,
    concurrency: int,
    delay: float,
    retries: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    db_path = Path(f"data/{domain}.duckdb")
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    conn = duckdb.connect(str(db_path))
    try:
        records = _load_null_records(conn, limit)
        logger.info(
            f"[{domain}] {len(records)} vacancies with null description"
            f"{' (limited)' if limit else ''}"
        )
        if not records:
            return

        sem = asyncio.Semaphore(concurrency)
        limiter = RateLimiter(delay)
        async with httpx.AsyncClient(
            base_url=HH_SITE_HOST,
            headers=_HEADERS,
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            total_filled = 0
            for attempt in range(retries + 1):
                if not records:
                    break
                logger.info(f"Pass {attempt + 1}/{retries + 1}: {len(records)} to fetch")
                filled, errors = await _run_pass(
                    client, sem, limiter, conn, records, dry_run
                )
                total_filled += filled
                logger.info(
                    f"  Pass {attempt + 1}: filled {filled}, "
                    f"errors: {errors if errors else 'none'}"
                )
                if dry_run:
                    break
                # Refresh the remaining-null set for the next pass.
                records = _load_null_records(conn, None)
                if records and attempt < retries:
                    pause = 15.0 * (attempt + 1)
                    logger.info(f"  Pausing {pause:.0f}s before retry pass")
                    await asyncio.sleep(pause)

        logger.info(f"[{domain}] total filled: {total_filled}")
        if not dry_run:
            remaining = len(_load_null_records(conn, None))
            logger.info(f"[{domain}] remaining null: {remaining}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill null vacancy descriptions")
    parser.add_argument("domain", help="Domain name (physics/biology/chemistry)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        _backfill(
            args.domain,
            args.concurrency,
            args.delay,
            args.retries,
            args.limit,
            args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
