"""Deterministic HTML parsers for LinkedIn Wayback Machine job pages.

Two era-specific entry points are provided:

* :func:`parse_legacy_linkedin` — 2013–2016 ``/jobs2/view/<id>`` pages.
* :func:`parse_modern_linkedin` — 2019+ ``/jobs/view/<slug>-<jobid>`` pages.

Both extract structured fields mapped to hh.ru API JSON field names.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from krm.phase0.parsers.common import (
    SalaryDict,
    clean_html,
    extract_jsonld,
    first_text,
    make_empty_result,
    parse_iso_date,
    parse_salary_from_text,
)

if TYPE_CHECKING:
    from bs4 import Tag

_LEGACY_PARSER_VERSION: str = "linkedin_legacy_v1"
_MODERN_PARSER_VERSION: str = "linkedin_modern_v1"


def _parse_job_criteria(criteria_section: Tag) -> dict[str, str | None]:
    """Parse the legacy LinkedIn job criteria 'Primary' section.

    Extracts: experience (seniority level), employment type, job function, industries.
    """
    result: dict[str, str | None] = {
        "seniority": None,
        "employment_type": None,
        "job_function": None,
        "industry": None,
    }

    # Legacy pages use dt/dd pairs or labeled spans within job-criteria
    items = criteria_section.find_all(["dt", "dd", "span", "div", "li"])
    label_map = {
        "seniority level": "seniority",
        "seniority": "seniority",
        "experience": "seniority",
        "experience level": "seniority",
        "employment type": "employment_type",
        "job function": "job_function",
        "functions": "job_function",
        "industry": "industry",
        "industries": "industry",
    }

    for item in items:
        text = item.get_text(strip=True)
        if not text:
            continue
        text_lower = text.lower().rstrip(":")
        for label, key in label_map.items():
            if text_lower == label and result[key] is None:
                next_el = item.find_next_sibling(["dd", "span", "div", "li"])
                if next_el is not None:
                    val = clean_html(next_el.get_text())
                    if val:
                        result[key] = val
                break

    # Try finding spans with class containing 'criteria' — look for label/value pairs
    criteria_spans = criteria_section.select("span[class*='criteria']")
    for span in criteria_spans:
        text = clean_html(span.get_text())
        if text is None:
            continue
        text_lower = text.lower().rstrip(":")
        for label, key in label_map.items():
            if text_lower == label and result[key] is None:
                next_span = span.find_next_sibling("span")
                if next_span is not None:
                    val = clean_html(next_span.get_text())
                    if val:
                        result[key] = val
                break

    return result


def _extract_skills(soup: Tag) -> list[str]:
    """Extract key skills from legacy LinkedIn job pages.

    Looks for:
    - Elements with class job-criteria__text
    - Sections labeled 'Skills' or 'Qualifications'
    - List items under those sections
    """
    skills: list[str] = []

    # Method 1: job-criteria__text spans
    skill_spans = soup.select("span.job-criteria__text")
    for span in skill_spans:
        text = clean_html(span.get_text())
        if text is not None and text not in skills:
            skills.append(text)

    # Method 2: Sections labeled "Skills" or "Qualifications"
    for header in soup.find_all(["h3", "h4", "strong", "b", "span", "div"]):
        header_text = clean_html(header.get_text())
        if header_text is None:
            continue
        header_lower = header_text.lower()
        if any(label in header_lower for label in ("skills", "qualifications", "requirements")):
            # Find the next list or container
            container = header.find_next(["ul", "ol", "div"])
            if container is not None:
                for li in container.find_all("li"):
                    text = clean_html(li.get_text())
                    if text is not None and text not in skills:
                        skills.append(text)

    return skills


def parse_legacy_linkedin(html: str, url: str) -> dict[str, object]:  # noqa: DICT_OK
    """Parse a 2013-2016 LinkedIn /jobs2/view/<id> Wayback Machine capture.

    Args:
        html: Raw HTML content of the page.
        url: The original URL of the page (for metadata).

    Returns:
        dict with hh.ru API-compatible JSON field names.
    """
    soup: Tag = BeautifulSoup(html, "lxml")

    # Detect login-gated page — if there's a sign-in prompt and no job content
    login_indicators = soup.select(
        "div.sign-in-modal, div.login-form, div.reg-upsell, "
        + "form.login, h1:contains('Sign in'), h2:contains('Sign in')"
    )
    title_el = soup.select_one("h1.title, h2.jobs-title, h1.job-title, h1[class*='title']")
    if login_indicators and title_el is None:
        return make_empty_result(_LEGACY_PARSER_VERSION, url)

    # Job title
    name = first_text(
        soup,
        ["h1.title", "h2.jobs-title", "h1.job-title", "h1[class*='title']"],
    )

    # Description
    description = first_text(
        soup,
        [
            "div.description",
            "div.job-description",
            "section.description",
            "div[class*='description']",
        ],
    )

    # Employer name
    employer_name = first_text(
        soup,
        [
            "a.company-name-link",
            "span.company",
            "a[data-tn-element='companyName']",
            "a[class*='company']",
        ],
    )

    # Location / area
    area_name = first_text(
        soup,
        [
            "span.location",
            "span.job-location",
            "div.location",
            "span[class*='location']",
        ],
    )

    # Published date
    published_at: str | None = None
    posted_el = soup.select_one("span.posted-time, span[class*='posted'], time[datetime]")
    if posted_el is not None:
        dt_attr = posted_el.get("datetime")
        if isinstance(dt_attr, str):
            published_at = parse_iso_date(dt_attr)
        if published_at is None:
            posted_text = clean_html(posted_el.get_text())
            if posted_text is not None:
                posted_text_lower = posted_text.lower()
                if "posted" in posted_text_lower:
                    # Try ISO date nearby
                    date_match = re.search(r"\d{4}-\d{2}-\d{2}", posted_text)
                    if date_match is not None:
                        published_at = parse_iso_date(date_match.group())
                    else:
                        published_at = posted_text

    # Salary from description
    salary: SalaryDict = {"from": None, "to": None, "currency": None}
    if description is not None:
        salary = parse_salary_from_text(description)

    # Job criteria section
    criteria_section = soup.select_one(
        "div.job-criteria, section.job-criteria, "
        + "div[class*='job-criteria'], div[class*='criteria']"
    )
    experience_name: str | None = None
    industry: str | None = None

    if criteria_section is not None:
        criteria = _parse_job_criteria(criteria_section)
        experience_name = criteria.get("seniority")
        industry = criteria.get("industry")

    # Fallback: scan text for "Seniority level: X" or "Experience: X"
    if experience_name is None and description is not None:
        exp_match = re.search(
            r"(?:Seniority\s*level|Experience\s*level|Seniority)[:\s-]+([^\n,]{3,40})",
            description,
            re.IGNORECASE,
        )
        if exp_match is not None:
            experience_name = exp_match.group(1).strip()

    # Fallback: scan text for "Industry: X"
    if industry is None and description is not None:
        ind_match = re.search(
            r"(?:Industry|Industries)[:\s-]+([^\n,]{3,60})",
            description,
            re.IGNORECASE,
        )
        if ind_match is not None:
            industry = ind_match.group(1).strip()

    # Skills
    key_skills = _extract_skills(soup)

    return {
        "name": name,
        "description": description,
        "employer": {"name": employer_name},
        "area": {"name": area_name},
        "salary": salary,
        "published_at": published_at,
        "key_skills": key_skills,
        "experience": {"name": experience_name},
        "industry": industry,
        "professional_roles": [],
        "_parser_version": _LEGACY_PARSER_VERSION,
        "_source_url": url,
    }


def _parse_jsonld(soup: Tag) -> dict[str, object] | None:
    """Extract the ``JobPosting`` JSON-LD object from a modern LinkedIn page.

    Modern pages (2019+) often embed structured ``JobPosting`` data.
    """
    data = extract_jsonld(soup)
    if data is not None and data.get("@type") == "JobPosting":
        return data
    return None


def _extract_skills_from_section(soup: Tag) -> list[str]:
    """Extract skills from 'Desired Skills and Experience' block in modern pages.

    Looks for section headers containing 'Skills' and extracts list items.
    """
    skills: list[str] = []

    for header in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
        header_text = clean_html(header.get_text())
        if header_text is None:
            continue
        header_lower = header_text.lower()
        if "skill" not in header_lower:
            continue

        # Find the nearest container with list items
        parent = header.parent
        if parent is None:
            continue

        # Search within parent and siblings for lists
        for container in [parent] + list(parent.find_next_siblings("div") or []):
            for li in container.find_all("li"):
                text = clean_html(li.get_text())
                if text is not None and text not in skills:
                    skills.append(text)

    return skills


def _extract_modern_criteria_items(soup: Tag) -> dict[str, str | None]:
    """Extract criteria items from modern LinkedIn job pages.

    Modern pages use elements like:
    - span.description__job-criteria-text
    - span.job-criteria__text
    - li.description__job-criteria-item
    """
    result: dict[str, str | None] = {
        "seniority": None,
        "employment_type": None,
        "industry": None,
    }

    criteria_items = soup.select(
        "li.description__job-criteria-item, "
        + "span.description__job-criteria-text, "
        + "span.job-criteria__text"
    )

    label_patterns: list[tuple[str, str]] = [
        ("seniority", "seniority"),
        ("experience", "seniority"),
        ("employment", "employment_type"),
        ("industry", "industry"),
    ]

    for item in criteria_items:
        text = clean_html(item.get_text())
        if text is None:
            continue
        text_lower = text.lower()

        # Check for label: value pattern within the element
        for label_word, key in label_patterns:
            if label_word in text_lower and result[key] is None:
                # Try to extract the value after the label
                parts = re.split(r"[:–\n]", text, maxsplit=1)
                if len(parts) > 1:
                    val = clean_html(parts[1])
                    if val:
                        result[key] = val
                break

    return result


def parse_modern_linkedin(html: str, url: str) -> dict[str, object]:  # noqa: DICT_OK
    """Parse a 2019+ LinkedIn /jobs/view/<slug>-<jobid> Wayback Machine capture.

    Args:
        html: Raw HTML content of the page.
        url: The original URL of the page (for metadata).

    Returns:
        dict with hh.ru API-compatible JSON field names.
    """
    soup: Tag = BeautifulSoup(html, "lxml")

    # Detect login-gated page
    login_indicators = soup.select(
        "div.sign-in-modal, div.login-form, div.reg-upsell, "
        + "form.login, nav#global-nav[class*='signin']"
    )
    title_el = soup.select_one(
        "h1.topcard__title, h1.top-card-layout__title, h1[class*='title']"
    )
    if login_indicators and title_el is None:
        return make_empty_result(_MODERN_PARSER_VERSION, url)

    # ── Job title ──
    name = first_text(
        soup,
        [
            "h1.topcard__title",
            "h1.top-card-layout__title",
            "h1[class*='title']",
        ],
    )

    # ── Description ──
    description = first_text(
        soup,
        [
            "div.description__text",
            "div.show-more-less-html__markup",
            "section.description",
            "div[class*='description']",
        ],
    )

    # ── Employer ──
    employer_name = first_text(
        soup,
        [
            "a.topcard__org-name-link",
            "span.topcard__flavor",
            "a[class*='org-name']",
            "span[class*='company']",
        ],
    )

    # ── Location ──
    area_name = first_text(
        soup,
        [
            "span.topcard__flavor--bullet",
            "span[class*='location']",
        ],
    )

    # ── Published date ──
    published_at: str | None = None
    posted_el = soup.select_one(
        "span.posted-time-ago__text, "
        + "span[class*='time-ago'], "
        + "span[class*='posted-time'], "
        + "time[datetime]"
    )
    if posted_el is not None:
        dt_attr = posted_el.get("datetime")
        if isinstance(dt_attr, str):
            published_at = parse_iso_date(dt_attr)
        if published_at is None:
            posted_text = clean_html(posted_el.get_text())
            if posted_text is not None:
                published_at = posted_text

    # ── JSON-LD fallback ──
    jsonld = _parse_jsonld(soup)
    if jsonld is not None:
        if name is None:
            title_raw = jsonld.get("title")
            if isinstance(title_raw, str):
                name = title_raw
        if description is None:
            desc_raw = jsonld.get("description")
            if isinstance(desc_raw, str):
                description = clean_html(desc_raw)
        if employer_name is None:
            hiring_org = jsonld.get("hiringOrganization", {})
            if isinstance(hiring_org, dict):
                org_name = hiring_org.get("name")
                if isinstance(org_name, str):
                    employer_name = org_name
        if published_at is None:
            date_str = jsonld.get("datePosted")
            if isinstance(date_str, str):
                published_at = parse_iso_date(date_str)

    # ── Salary from description ──
    salary: SalaryDict = {"from": None, "to": None, "currency": None}
    # Also check JSON-LD for salary
    if jsonld is not None:
        base_salary = jsonld.get("baseSalary", {})
        if isinstance(base_salary, dict):
            sal_value = base_salary.get("value", {})
            if isinstance(sal_value, dict):
                min_val = sal_value.get("minValue")
                max_val = sal_value.get("maxValue")
                if isinstance(min_val, (int, float)):
                    salary["from"] = round(min_val)
                if isinstance(max_val, (int, float)):
                    salary["to"] = round(max_val)
            currency_val = base_salary.get("currency")
            if isinstance(currency_val, str):
                salary["currency"] = currency_val
    # Fallback: parse from description text
    if salary["from"] is None and description is not None:
        salary = parse_salary_from_text(description)

    # ── Criteria items ──
    criteria = _extract_modern_criteria_items(soup)
    experience_name = criteria.get("seniority")
    industry: str | None = criteria.get("industry")

    # Fallback: scan description for experience/industry labels
    if experience_name is None and description is not None:
        exp_match = re.search(
            r"(?:Seniority\s*level|Experience\s*level|Seniority)[:\s-]+([^\n,]{3,40})",
            description,
            re.IGNORECASE,
        )
        if exp_match is not None:
            experience_name = exp_match.group(1).strip()

    if industry is None and description is not None:
        ind_match = re.search(
            r"(?:Industry|Industries)[:\s-]+([^\n,]{3,60})",
            description,
            re.IGNORECASE,
        )
        if ind_match is not None:
            industry = ind_match.group(1).strip()

    # ── Skills ──
    key_skills = _extract_skills_from_section(soup)

    return {
        "name": name,
        "description": description,
        "employer": {"name": employer_name},
        "area": {"name": area_name},
        "salary": salary,
        "published_at": published_at,
        "key_skills": key_skills,
        "experience": {"name": experience_name},
        "industry": industry,
        "professional_roles": [],
        "_parser_version": _MODERN_PARSER_VERSION,
        "_source_url": url,
    }
