"""Phase 1 (site): HH.ru vacancy collection via the public website — no API key.

The hh.ru API host (``api.hh.ru``) is IP-blocked from some networks, but the
public website (``hh.ru``) is server-rendered and embeds the full vacancy data
as an HTML-escaped JSON blob inside ``<template id="HH-Lux-InitialState">``.
This module extracts that blob and maps it onto the hh.ru API JSON shape so the
rest of the pipeline consumes it unchanged.

Flow per keyword:
    1. GET /search/vacancy?text={kw}&page={n}  → vacancy list (id + basic fields)
    2. GET /vacancy/{id}                       → full description + key skills

Usage:
    from krm.phase_1_site import collect_site
    new_vacancies = collect_site(config)
"""

from __future__ import annotations

import asyncio
import html as htmlmod
import json
import re
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from krm.config import Config
from krm.lib.io import get_connection, init_tables, upsert_raw_vacancy
from krm.phase_1_collect import (
    _finalize_scrape_run,
    _insert_scrape_run,
    _make_run_id,
)

HH_SITE_HOST = "https://hh.ru"
MAX_PAGES = 1000  # safety ceiling only; normal stop is empty results / disabled paging

_INITIAL_STATE_RE = re.compile(
    r'<template[^>]*id="HH-Lux-InitialState"[^>]*>(.*?)</template>', re.DOTALL
)

# workExperience code → hh.ru API experience display name.
_EXPERIENCE_NAMES: dict[str, str] = {
    "noExperience": "Нет опыта",
    "between1And3": "От 1 года до 3 лет",
    "between3And6": "От 3 до 6 лет",
    "moreThan6": "Более 6 лет",
}


def _strip_html(text: str | None) -> str | None:
    if not text:
        return None
    return BeautifulSoup(text, "lxml").get_text(separator=" ", strip=True) or None


def _map_salary(compensation: dict[str, Any] | None) -> dict[str, int | None | str]:
    if not compensation:
        return {"from": None, "to": None, "currency": None}
    return {
        "from": compensation.get("from"),
        "to": compensation.get("to"),
        "currency": compensation.get("currencyCode"),
    }


def _published_at(raw: dict[str, Any]) -> str | None:
    """Extract publication timestamp — a plain string on the detail view, an
    ``{'$': iso}`` dict on the search list."""
    value = raw.get("publicationDate") or raw.get("publicationTime")
    if isinstance(value, dict):
        return value.get("$")
    return value


def _map_vacancy(raw: dict[str, Any], vacancy_id: str) -> dict[str, Any]:
    """Map a website vacancy view onto the hh.ru API JSON shape."""
    company = raw.get("company") or {}
    area = raw.get("area") or {}
    return {
        "id": str(vacancy_id),
        "name": raw.get("name"),
        "description": _strip_html(raw.get("description")),
        "employer": {"name": company.get("name") if isinstance(company, dict) else None},
        "area": {"name": area.get("name") if isinstance(area, dict) else None},
        "salary": _map_salary(raw.get("compensation")),
        "published_at": _published_at(raw),
        "key_skills": [
            {"name": s} for s in (raw.get("keySkills") or {}).get("keySkill", [])
        ],
        "experience": {
            "name": _EXPERIENCE_NAMES.get(raw.get("workExperience") or "")
        },
        "professional_roles": [
            {"id": rid, "name": ""} for rid in (raw.get("professionalRoleIds") or [])
        ],
    }


def _should_exclude(name: str | None, exclude_terms: list[str]) -> bool:
    """True if the vacancy title contains any exclusion term (case-insensitive)."""
    if not name or not exclude_terms:
        return False
    low = name.lower()
    return any(term in low for term in exclude_terms)


async def _fetch_state(
    client: httpx.AsyncClient, url: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Fetch a hh.ru page and return its parsed HH-Lux-InitialState JSON."""
    response = await client.get(url, params=params)
    response.raise_for_status()
    match = _INITIAL_STATE_RE.search(response.text)
    if match is None:
        raise RuntimeError(f"HH-Lux-InitialState not found in {url}")
    return json.loads(htmlmod.unescape(match.group(1)))


async def _collect_keyword_site(
    client: httpx.AsyncClient,
    conn: Any,
    run_id: str,
    keyword: str,
    min_interval: float,
    exclude_terms: list[str] | None = None,
) -> int:
    """Search + fetch detail for one keyword; store mapped vacancies."""
    exclude_terms = exclude_terms or []
    collected = 0
    page = 0
    while page < MAX_PAGES:
        await asyncio.sleep(min_interval)
        try:
            state = await _fetch_state(
                client,
                f"{HH_SITE_HOST}/search/vacancy",
                {"text": keyword, "page": page},
            )
        except Exception as exc:  # noqa: BLE001 — scrape boundary
            logger.warning(f"  search failed for {keyword!r} page {page}: {exc}")
            break

        search = state.get("vacancySearchResult", {})
        items = search.get("vacancies", [])
        if not items:
            break
        logger.info(
            f"  [{keyword}] page {page}: {len(items)} vacancies "
            f"(total {search.get('totalResults', '?')})"
        )

        for item in items:
            vacancy_id = str(item.get("vacancyId"))
            await asyncio.sleep(min_interval)
            try:
                detail = await _fetch_state(
                    client, f"{HH_SITE_HOST}/vacancy/{vacancy_id}", {}
                )
                mapped = _map_vacancy(detail.get("vacancyView", item), vacancy_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"  detail failed for {vacancy_id}: {exc}")
                mapped = _map_vacancy(item, vacancy_id)

            if _should_exclude(mapped.get("name"), exclude_terms):
                continue

            upsert_raw_vacancy(conn, run_id, vacancy_id, mapped)
            collected += 1

        nxt = (search.get("paging") or {}).get("next")
        if nxt and nxt.get("disabled"):
            break
        page += 1

    return collected


async def _collect_async(
    config: Config,
    conn: Any,
    keywords: list[str],
    exclude_terms: list[str],
) -> int:
    min_interval = 1.0 / max(config.rate_limit_rps, 1)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9",
    }
    async with httpx.AsyncClient(
        base_url=HH_SITE_HOST, headers=headers, timeout=30.0, follow_redirects=True
    ) as client:
        total = 0
        for keyword in keywords:
            run_id = _make_run_id(keyword)
            logger.info(f"Collecting keyword {keyword!r} (run {run_id})")
            _insert_scrape_run(conn, run_id, keyword, config)
            n = await _collect_keyword_site(
                client, conn, run_id, keyword, min_interval, exclude_terms
            )
            _finalize_scrape_run(conn, run_id, n)
            total += n
            logger.info(f"  [{keyword}] Collected {n} vacancies")
    return total


def collect_site(config: Config, domain: str | None = None) -> int:
    """Collect hh.ru vacancies via the public website (no API key).

    When ``domain`` is given (physics/biology/chemistry), collects that domain's
    keywords into its own database; otherwise collects ``config.keywords`` into
    the default database.
    """
    if domain is not None:
        db_path: Path | str = config.domain_db_path(domain)
        keywords = config.domain_keywords(domain)
        logger.info(
            f"Domain '{config.domain_name(domain)}' → {db_path} "
            f"({len(keywords)} keywords)"
        )
    else:
        db_path = "data/krm.duckdb"
        keywords = config.keywords

    conn = get_connection(db_path)
    init_tables(conn)
    try:
        total = asyncio.run(
            _collect_async(config, conn, keywords, config.exclude_terms)
        )
    finally:
        conn.close()
    return total
