"""Deterministic HTML parsers for hh.ru Wayback Machine vacancy pages.

Two era-specific entry points are provided:

* :func:`parse_legacy_vacancy` — 2010–2011 pages whose URLs look like
  ``.../vacancy.do?id=NNNNN``.
* :func:`parse_modern_vacancy` — 2012+ pages at ``/vacancy/<id>`` using
  ``data-qa`` attributes as stable selectors.

Both return a structured dict whose keys match the hh.ru API JSON field
names so downstream phases treat legacy and modern extractions uniformly.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from krm.phase0.parsers.common import (
    AreaDict,
    EmployerDict,
    NamedDict,
    SalaryDict,
    SkillDict,
    VacancyDict,
    extract_jsonld,
    first_text,
    parse_iso_date,
    parse_salary_from_text,
    str_or_none,
)

_LEGACY_PARSER_VERSION: str = "hhru_legacy_v1"
_MODERN_PARSER_VERSION: str = "hhru_modern_v1"


def parse_legacy_vacancy(html: str, _url: str) -> VacancyDict:  # noqa: DICT_OK — TypedDict usage
    """Parse a 2010–2011 hh.ru vacancy page (``.do`` URL) into structured data.

    Args:
        html: Raw HTML of the Wayback Machine capture.
        url: Original URL for reference (included in result).

    Returns:
        A ``VacancyDict`` with hh.ru API-compatible field names.  Every
        field is present; missing data is ``None`` rather than absent.
    """
    soup = BeautifulSoup(html, "lxml")

    name = first_text(soup, [
        "div.vacancy-title",
        "h1",
    ])

    description = first_text(soup, [
        "div.vacancy-description",
        "div.vacancy-content",
    ])

    employer_name = first_text(soup, [
        "span.company-name",
        "a.company-link",
        "div.company-name",
    ])
    employer: EmployerDict = {"name": employer_name}

    area_name = first_text(soup, [
        "span.location",
        "div.location",
    ])
    area: AreaDict = {"name": area_name}

    salary_text = first_text(soup, ["span.salary"])
    salary: SalaryDict = parse_salary_from_text(salary_text)

    raw_date = first_text(soup, [
        "span.vacancy-creation-date",
        "div.vacancy-creation-date",
    ])
    if raw_date is None:
        meta_el = soup.select_one("meta[name='DC.date']")
        if meta_el is not None:
            raw_date = str_or_none(meta_el.get("content"))
    published_at = parse_iso_date(raw_date)

    exp_name = first_text(soup, [
        "span.experience",
        "div.experience",
    ])
    experience: NamedDict = {"name": exp_name}

    key_skills: list[SkillDict] = []

    return {
        "name": name,
        "description": description,
        "employer": employer,
        "area": area,
        "salary": salary,
        "published_at": published_at,
        "experience": experience,
        "key_skills": key_skills,
        "professional_roles": [],
        "_parser_version": _LEGACY_PARSER_VERSION,
    }


def parse_modern_vacancy(html: str, _url: str) -> VacancyDict:  # noqa: DICT_OK — TypedDict usage
    """Parse a 2012+ hh.ru vacancy page (``/vacancy/<id>``) into structured data.

    Uses ``data-qa`` attributes as stable selectors as recommended by
    hh.ru's scraping guidelines.

    Args:
        html: Raw HTML of the Wayback Machine capture.
        url: Original URL for reference (included in result).

    Returns:
        A ``VacancyDict`` with hh.ru API-compatible field names.  Every
        field is present; missing data is ``None`` rather than absent.
    """
    soup = BeautifulSoup(html, "lxml")

    name = first_text(soup, ['h1[data-qa="vacancy-title"]'])

    description = first_text(soup, [
        'div[data-qa="vacancy-description"]',
        'div.vacancy-branded-user-content',
    ])
    if description is None:
        ld_data = extract_jsonld(soup)
        if ld_data is not None:
            ld_desc = ld_data.get("description")
            if isinstance(ld_desc, str) and ld_desc:
                description = BeautifulSoup(ld_desc, "lxml").get_text(
                    separator="\n", strip=True
                )
    if description is None:
        guc_divs = soup.select('div.g-user-content, div[class*="g-user-content"]')
        if guc_divs:
            description = "\n".join(d.get_text(strip=True) for d in guc_divs)

    employer_name = first_text(soup, [
        'a[data-qa="vacancy-company-name"]',
        'span[data-qa="vacancy-company-name"]',
        'div[data-qa="vacancy-company-name"]',
    ])
    employer: EmployerDict = {"name": employer_name}

    area_name = first_text(soup, [
        'p[data-qa="vacancy-view-location"]',
        'span[data-qa="vacancy-view-raw-address"]',
    ])
    area: AreaDict = {"name": area_name}

    salary_text = first_text(soup, [
        'span[data-qa="vacancy-salary"]',
        "p.vacancy-salary",
    ])
    salary: SalaryDict = parse_salary_from_text(salary_text)

    raw_date = first_text(soup, ["p.vacancy-creation-time"])
    if raw_date is None:
        meta_el = soup.select_one('meta[itemprop="datePosted"]')
        if meta_el is not None:
            raw_date = str_or_none(meta_el.get("content"))
    if raw_date is None:
        ld_data = extract_jsonld(soup)
        if ld_data is not None:
            date_posted = ld_data.get("datePosted")
            if isinstance(date_posted, str):
                raw_date = date_posted
    published_at = parse_iso_date(raw_date)

    exp_name = first_text(soup, ['span[data-qa="vacancy-experience"]'])
    experience: NamedDict = {"name": exp_name}

    skill_els = soup.select('[data-qa="skills-element"]')
    key_skills: list[SkillDict] = [
        {"name": el.get_text(strip=True)} for el in skill_els
    ]

    schedule_name = first_text(soup, [
        'p[data-qa="vacancy-view-employment-mode"]',
    ])
    schedule: NamedDict = {"name": schedule_name}

    return {
        "name": name,
        "description": description,
        "employer": employer,
        "area": area,
        "salary": salary,
        "published_at": published_at,
        "experience": experience,
        "key_skills": key_skills,
        "professional_roles": [],
        "schedule": schedule,
        "_parser_version": _MODERN_PARSER_VERSION,
    }
