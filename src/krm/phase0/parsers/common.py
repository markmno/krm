"""Shared extraction helpers and type definitions for the Phase 0 HTML parsers."""

from __future__ import annotations

import json
import re
from typing import TypedDict

from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# TypedDicts matching hh.ru API JSON field names
# ---------------------------------------------------------------------------


class EmployerDict(TypedDict):
    """Employer info extracted from vacancy HTML."""

    name: str | None


class AreaDict(TypedDict):
    """Location info extracted from vacancy HTML."""

    name: str | None


# ``from`` is a Python keyword, so ``SalaryDict`` is declared with the
# functional ``TypedDict`` syntax (the key names match the hh.ru API JSON).
SalaryDict = TypedDict(
    "SalaryDict",
    {"from": int | None, "to": int | None, "currency": str | None},
)
SalaryDict.__doc__ = "Salary range extracted from vacancy HTML (hh.ru API field names)."


class SkillDict(TypedDict):
    """A single key skill."""

    name: str


class NamedDict(TypedDict):
    """Generic named entity (experience, employment, schedule)."""

    name: str | None


class VacancyDict(TypedDict, total=False):
    """Structured vacancy data matching hh.ru API JSON field names.

    Fields are always present but may be ``None``.  ``schedule`` is
    modern-only, ``key_skills`` may be empty in legacy pages.
    """

    name: str | None
    description: str | None
    employer: EmployerDict
    area: AreaDict
    salary: SalaryDict
    published_at: str | None
    experience: NamedDict
    key_skills: list[SkillDict]
    professional_roles: list[dict[str, str]]
    schedule: NamedDict
    _parser_version: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

#: Russian genitive month names → two-digit month number.
_MONTHS_RU: dict[str, str] = {
    "января": "01", "февраля": "02", "марта": "03",
    "апреля": "04", "мая": "05", "июня": "06",
    "июля": "07", "августа": "08", "сентября": "09",
    "октября": "10", "ноября": "11", "декабря": "12",
}

_MONTHS_RU_PATTERN = "|".join(_MONTHS_RU)


def first_text(soup: BeautifulSoup | Tag, selectors: list[str]) -> str | None:
    """Try CSS selectors in order, returning the first non-empty stripped text.

    Args:
        soup: A ``BeautifulSoup`` document or ``Tag`` subtree to search within.
        selectors: List of CSS selector strings tried left-to-right.

    Returns:
        The stripped text of the first matching element with non-whitespace
        content, or ``None`` if no selector matches or every match is empty.
    """
    for sel in selectors:
        el = soup.select_one(sel)
        if el is None:
            continue
        text = el.get_text(strip=True)
        if text:
            return text
    return None


def clean_html(text: str | None) -> str | None:
    """Strip HTML tags and normalise whitespace from a string.

    Args:
        text: Raw string that may contain HTML markup.

    Returns:
        Tag-stripped, whitespace-collapsed string, or ``None`` if the
        input was ``None`` or produced only whitespace.
    """
    if text is None:
        return None
    # Remove HTML/XML tags
    stripped = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", stripped).strip()
    return cleaned or None


def str_or_none(value: str | list[str] | None) -> str | None:
    """Coerce a BS4 ``_AttributeValue`` to ``str | None``.

    BeautifulSoup's ``Tag.get()`` returns ``_AttributeValue | None``
    which can be a ``list[str]``.  This helper narrows it to a plain
    string or ``None`` for safe use with ``parse_iso_date`` and similar.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def parse_iso_date(text: str | None) -> str | None:
    """Normalise a date string to ISO 8601 (``YYYY-MM-DD``).

    Handles the most common hh.ru date formats:

    * ``"DD month YYYY"`` — Russian locale (``"12 января 2011"``)
    * ``"DD.MM.YYYY"``
    * ``YYYY-MM-DD`` — pass-through

    Args:
        text: Raw date string from the HTML.

    Returns:
        ISO 8601 date string, or ``None`` if input is ``None`` or
        unrecognised.
    """
    if text is None:
        return None

    cleaned = text.strip()

    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        return cleaned

    # DD.MM.YYYY
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", cleaned)
    if m:
        day, month, year = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # "DD month_ru YYYY"
    m = re.match(
        rf"(\d{{1,2}})\s+({_MONTHS_RU_PATTERN})\s+(\d{{4}})",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        day, month_ru, year = m.groups()
        month_num = _MONTHS_RU.get(month_ru.lower())
        if month_num is not None:
            return f"{year}-{month_num}-{int(day):02d}"

    return cleaned  # fallback: return as-is


# ---------------------------------------------------------------------------
# Salary extraction (shared by all four parsers)
# ---------------------------------------------------------------------------

_SALARY_RE = re.compile(
    r"(?:\$|USD\s*|€|EUR\s*|£|GBP\s*)?\s*"
    + r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[Kk]?)"
    + r"\s*(?:-|–|to|—)\s*"
    + r"(?:\$|USD\s*|€|EUR\s*|£|GBP\s*)?\s*"
    + r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[Kk]?)",
)

_SINGLE_SALARY_RE = re.compile(
    r"(?:\$|USD\s*|€|EUR\s*|£|GBP\s*)\s*"
    + r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[Kk]?)",
)

_SALARY_CURRENCY_MAP: dict[str, str] = {
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

_RUSSIAN_NUMBER_RE = re.compile(r"\d[\d\s\u00a0]*\d|\d+")


def _detect_salary_currency(text: str) -> str | None:
    text_lower = text.lower()
    if "руб" in text_lower:
        return "RUR"
    for symbol, code in _SALARY_CURRENCY_MAP.items():
        if symbol in text_lower:
            return code
    return None


def _is_russian_salary(text: str) -> bool:
    text_lower = text.lower()
    return "руб" in text_lower or "от" in text_lower or "до" in text_lower


def _parse_russian_salary(text: str, result: SalaryDict) -> SalaryDict:
    nums: list[int] = []
    for m in _RUSSIAN_NUMBER_RE.finditer(text):
        cleaned = m.group().replace(" ", "").replace("\xa0", "")
        if cleaned.isdigit():
            nums.append(int(cleaned))

    if not nums:
        return result

    text_lower = text.lower()
    if len(nums) == 1:
        if "от" in text_lower:
            result["from"] = nums[0]
        elif "до" in text_lower:
            result["to"] = nums[0]
        else:
            result["from"] = nums[0]
        return result

    nums.sort()
    result["from"] = nums[0]
    result["to"] = nums[-1]
    return result


def _parse_english_salary(text: str, result: SalaryDict) -> SalaryDict:
    text_lower = text.lower()

    def _parse_amount(raw: str) -> int | None:
        raw = raw.replace(",", "").strip()
        if raw.upper().endswith("K"):
            raw = raw[:-1].strip()
            multiplier = 1000
        else:
            multiplier = 1
        try:
            val = float(raw) * multiplier
            return round(val)
        except ValueError:
            return None

    m = _SALARY_RE.search(text)
    if m is not None:
        result["from"] = _parse_amount(m.group(1))
        result["to"] = _parse_amount(m.group(2))
        return result

    m = _SINGLE_SALARY_RE.search(text)
    if m is not None:
        val = _parse_amount(m.group(1))
        result["from"] = val
        result["to"] = val
        return result

    # "per hour" or "/hr" salary fragments annualised at 40h × 52w = 2080h
    if "per hour" in text_lower or "/hr" in text_lower or "/hour" in text_lower:
        hr_match = re.search(r"(\d{1,4}(?:,\d{3})*(?:\.\d+)?)", text)
        if hr_match is not None:
            val = _parse_amount(hr_match.group(1))
            if val is not None:
                annual = val * 2080
                result["from"] = annual
                result["to"] = annual

    return result


def parse_salary_from_text(text: str | None) -> SalaryDict:
    """Extract a salary range from free text, in Russian or English.

    Russian forms (``"от 100 000 до 150 000 руб."``, ``"до X"``, ``"от X"``)
    and English forms (``"$80,000 - $120,000"``, a single value, ``"K"``
    suffix, ``"per hour"``/``"/hr"`` annualised at ×2080) are both handled.

    Returns a ``SalaryDict`` with keys ``from``, ``to``, ``currency`` matching
    the hh.ru API JSON field names; missing pieces are ``None``.
    """
    result: SalaryDict = {"from": None, "to": None, "currency": None}
    if text is None:
        return result

    text = text.strip()
    if not text:
        return result

    result["currency"] = _detect_salary_currency(text)

    if _is_russian_salary(text):
        return _parse_russian_salary(text, result)
    return _parse_english_salary(text, result)


# ---------------------------------------------------------------------------
# JSON-LD extraction (shared by hh.ru modern & LinkedIn modern parsers)
# ---------------------------------------------------------------------------


def extract_jsonld(soup: BeautifulSoup | Tag) -> dict[str, object] | None:
    """Return the first JSON-LD dict found on a page.

    Handles both a single JSON object and a JSON array of objects in the
    first ``script[type="application/ld+json"]`` tag.  Returns ``None`` if
    no parseable JSON-LD object exists.  Callers filter by
    ``@type == "JobPosting"`` themselves when they need a specific entity.
    """
    scripts = soup.select("script[type='application/ld+json']")
    for script in scripts:
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if isinstance(item, dict):
                return item
    return None


# ---------------------------------------------------------------------------
# Empty-result skeleton (shared by LinkedIn legacy & modern parsers)
# ---------------------------------------------------------------------------


def make_empty_result(parser_version: str, url: str) -> dict[str, object]:
    """Build an all-None parse result skeleton for a login-gated/empty page."""
    return {
        "name": None,
        "description": None,
        "employer": {"name": None},
        "area": {"name": None},
        "salary": {"from": None, "to": None, "currency": None},
        "published_at": None,
        "key_skills": [],
        "experience": {"name": None},
        "industry": None,
        "professional_roles": [],
        "_parser_version": parser_version,
        "_source_url": url,
    }
