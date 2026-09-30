"""Tests for deterministic hh.ru Wayback Machine HTML parsers.

Coverage:
* Full-field legacy (.do) and modern (/vacancy/<id>) parsing
* Missing fields → ``None`` (never ``KeyError``)
* Empty HTML → dict of ``None`` defaults
* Truncated / partial HTML → graceful degradation
* ``key_skills`` shape: ``[{"name": str}]``
* ``employer.name`` and ``area.name`` extraction
* Salary parsing: ``from``, ``to``, ``currency``
* Date normalisation: Russian locale, ``DD.MM.YYYY``, ISO pass-through
* ``_parser_version`` marker
"""

from __future__ import annotations

import pytest

from krm.phase0.parsers.hhru import parse_legacy_vacancy, parse_modern_vacancy


# ---------------------------------------------------------------------------
# Synthetic HTML fixtures
# ---------------------------------------------------------------------------

LEGACY_FULL_HTML = """\
<html><body>
<div class="vacancy-title">Инженер-программист</div>
<div class="vacancy-description">
Разработка и поддержка ПО, участие в проектах.
</div>
<span class="company-name">ООО Ромашка</span>
<span class="location">Москва</span>
<span class="salary">от 100 000 до 150 000 руб.</span>
<span class="vacancy-creation-date">12 января 2011</span>
<span class="experience">1–3 года</span>
</body></html>
"""

LEGACY_MINIMAL_HTML = """\
<html><body>
<div class="vacancy-title">Тестировщик</div>
</body></html>
"""

LEGACY_EMPTY_HTML = """\
<html><body></body></html>
"""

LEGACY_TRUNCATED_HTML = """\
<html><body>
<div class="vacancy-title">Инженер
"""

MODERN_FULL_HTML = """\
<html><body>
<h1 data-qa="vacancy-title">Python-разработчик</h1>
<div data-qa="vacancy-description">Разработка микросервисов на Python, PostgreSQL.</div>
<a data-qa="vacancy-company-name">Яндекс</a>
<p data-qa="vacancy-view-location">Санкт-Петербург</p>
<span data-qa="vacancy-salary">от 200 000 до 350 000 руб.</span>
<p class="vacancy-creation-time">15.06.2023</p>
<span data-qa="vacancy-experience">3–6 лет</span>
<span data-qa="skills-element">Python</span>
<span data-qa="skills-element">PostgreSQL</span>
<span data-qa="skills-element">Docker</span>
<p data-qa="vacancy-view-employment-mode">Полный день</p>
</body></html>
"""

MODERN_MINIMAL_HTML = """\
<html><body>
<h1 data-qa="vacancy-title">Аналитик</h1>
</body></html>
"""

MODERN_EMPTY_HTML = """\
<html><body></body></html>
"""

MODERN_TRUNCATED_HTML = """\
<html><body>
<h1 data-qa="vacancy-title">Разработчик
"""

URL = "https://web.archive.org/web/20110615000000/https://hh.ru/vacancy.do?id=12345"


# ---------------------------------------------------------------------------
# Legacy parser tests
# ---------------------------------------------------------------------------

class TestParseLegacyVacancy:
    """Given: legacy hh.ru HTML (2010–2011, .do URLs)."""

    def test_full_html_extracts_all_fields(self) -> None:
        """When: HTML contains every field, Then: all values are extracted correctly."""
        result = parse_legacy_vacancy(LEGACY_FULL_HTML, URL)

        assert result["name"] == "Инженер-программист"
        assert result["description"] == "Разработка и поддержка ПО, участие в проектах."
        assert result["employer"]["name"] == "ООО Ромашка"
        assert result["area"]["name"] == "Москва"
        assert result["salary"]["from"] == 100_000
        assert result["salary"]["to"] == 150_000
        assert result["salary"]["currency"] == "RUR"
        assert result["published_at"] == "2011-01-12"
        assert result["experience"]["name"] == "1–3 года"
        assert result["key_skills"] == []
        assert result["professional_roles"] == []
        assert result["_parser_version"] == "hhru_legacy_v1"

    def test_minimal_html_has_none_fallback(self) -> None:
        """When: HTML has only the title, Then: missing fields are None, not absent."""
        result = parse_legacy_vacancy(LEGACY_MINIMAL_HTML, URL)

        assert result["name"] == "Тестировщик"
        assert result["description"] is None
        assert result["employer"]["name"] is None
        assert result["area"]["name"] is None
        assert result["salary"]["from"] is None
        assert result["salary"]["to"] is None
        assert result["salary"]["currency"] is None
        assert result["published_at"] is None
        assert result["experience"]["name"] is None

    def test_empty_html_returns_none_defaults(self) -> None:
        """When: HTML is empty, Then: every value is None."""
        result = parse_legacy_vacancy(LEGACY_EMPTY_HTML, URL)

        assert result["name"] is None
        assert result["description"] is None
        assert result["employer"]["name"] is None
        assert result["salary"]["from"] is None
        assert result["published_at"] is None
        assert result["_parser_version"] == "hhru_legacy_v1"

    def test_truncated_html_does_not_crash(self) -> None:
        """When: HTML is truncated mid-tag, Then: parser returns without raising."""
        result = parse_legacy_vacancy(LEGACY_TRUNCATED_HTML, URL)
        assert result["_parser_version"] == "hhru_legacy_v1"
        # bs4 recovers what it can
        assert isinstance(result["name"], str)

    def test_employer_name_extracted(self) -> None:
        """When: company-name span exists, Then: employer.name is populated."""
        html = '<span class="company-name">ЗАО Тест</span>'
        result = parse_legacy_vacancy(html, URL)
        assert result["employer"]["name"] == "ЗАО Тест"

    def test_area_name_extracted(self) -> None:
        """When: location span exists, Then: area.name is populated."""
        html = '<span class="location">Новосибирск</span>'
        result = parse_legacy_vacancy(html, URL)
        assert result["area"]["name"] == "Новосибирск"


# ---------------------------------------------------------------------------
# Modern parser tests
# ---------------------------------------------------------------------------

class TestParseModernVacancy:
    """Given: modern hh.ru HTML (2012+, /vacancy/<id>)."""

    def test_full_html_extracts_all_fields(self) -> None:
        """When: HTML contains every field, Then: all values are extracted correctly."""
        result = parse_modern_vacancy(MODERN_FULL_HTML, URL)

        assert result["name"] == "Python-разработчик"
        assert result["description"] == "Разработка микросервисов на Python, PostgreSQL."
        assert result["employer"]["name"] == "Яндекс"
        assert result["area"]["name"] == "Санкт-Петербург"
        assert result["salary"]["from"] == 200_000
        assert result["salary"]["to"] == 350_000
        assert result["salary"]["currency"] == "RUR"
        assert result["published_at"] == "2023-06-15"
        assert result["experience"]["name"] == "3–6 лет"
        assert result["schedule"]["name"] == "Полный день"
        assert result["professional_roles"] == []
        assert result["_parser_version"] == "hhru_modern_v1"

    def test_key_skills_shape(self) -> None:
        """When: skills-element elements exist, Then: result is list of {'name': str}."""
        result = parse_modern_vacancy(MODERN_FULL_HTML, URL)

        skills = result["key_skills"]
        assert len(skills) == 3
        assert skills[0] == {"name": "Python"}
        assert skills[1] == {"name": "PostgreSQL"}
        assert skills[2] == {"name": "Docker"}
        # Every element has exactly one key: "name"
        for s in skills:
            assert list(s.keys()) == ["name"]
            assert isinstance(s["name"], str)

    def test_minimal_html_has_none_fallback(self) -> None:
        """When: HTML has only the title, Then: missing fields are None, not absent."""
        result = parse_modern_vacancy(MODERN_MINIMAL_HTML, URL)

        assert result["name"] == "Аналитик"
        assert result["description"] is None
        assert result["employer"]["name"] is None
        assert result["area"]["name"] is None
        assert result["salary"]["from"] is None
        assert result["published_at"] is None
        assert result["experience"]["name"] is None
        assert result["schedule"]["name"] is None
        assert result["key_skills"] == []

    def test_empty_html_returns_none_defaults(self) -> None:
        """When: HTML is empty, Then: every value is None."""
        result = parse_modern_vacancy(MODERN_EMPTY_HTML, URL)

        assert result["name"] is None
        assert result["description"] is None
        assert result["employer"]["name"] is None
        assert result["salary"]["from"] is None
        assert result["published_at"] is None
        assert result["_parser_version"] == "hhru_modern_v1"

    def test_truncated_html_does_not_crash(self) -> None:
        """When: HTML is truncated mid-tag, Then: parser returns without raising."""
        result = parse_modern_vacancy(MODERN_TRUNCATED_HTML, URL)
        assert result["_parser_version"] == "hhru_modern_v1"
        assert isinstance(result["name"], str)

    def test_employer_name_via_company_name_link(self) -> None:
        """When: company-name link exists, Then: employer.name is populated."""
        html = '<a data-qa="vacancy-company-name">ООО Альфа</a>'
        result = parse_modern_vacancy(html, URL)
        assert result["employer"]["name"] == "ООО Альфа"

    def test_employer_name_via_company_name_span(self) -> None:
        """When: only span company-name exists, Then: employer.name falls back to span."""
        html = '<span data-qa="vacancy-company-name">ИП Бета</span>'
        result = parse_modern_vacancy(html, URL)
        assert result["employer"]["name"] == "ИП Бета"

    def test_area_name_via_location(self) -> None:
        """When: vacancy-view-location exists, Then: area.name is populated."""
        html = '<p data-qa="vacancy-view-location">Казань</p>'
        result = parse_modern_vacancy(html, URL)
        assert result["area"]["name"] == "Казань"

    def test_area_name_via_raw_address(self) -> None:
        """When: only raw-address span exists, Then: area.name falls back to raw addr."""
        html = '<span data-qa="vacancy-view-raw-address">ул. Пушкина, 10</span>'
        result = parse_modern_vacancy(html, URL)
        assert result["area"]["name"] == "ул. Пушкина, 10"

    def test_published_at_via_meta_dateposted(self) -> None:
        """When: no .vacancy-creation-time but meta[itemprop=datePosted] exists, Then: date extracted."""
        html = '<meta itemprop="datePosted" content="2022-03-14">'
        result = parse_modern_vacancy(html, URL)
        assert result["published_at"] == "2022-03-14"


# ---------------------------------------------------------------------------
# Salary parsing tests
# ---------------------------------------------------------------------------

class TestSalaryParsing:
    """Given: various salary string formats."""

    def test_fromonly(self) -> None:
        """When: salary is 'от X', Then: from set, to None."""
        html = '<h1 data-qa="vacancy-title">X</h1><span data-qa="vacancy-salary">от 80 000 руб.</span>'
        result = parse_modern_vacancy(html, URL)
        assert result["salary"] == {"from": 80_000, "to": None, "currency": "RUR"}

    def test_to_only(self) -> None:
        """When: salary is 'до X', Then: to set, from None."""
        html = '<h1 data-qa="vacancy-title">X</h1><span data-qa="vacancy-salary">до 50 000 руб.</span>'
        result = parse_modern_vacancy(html, URL)
        assert result["salary"] == {"from": None, "to": 50_000, "currency": "RUR"}

    def test_range(self) -> None:
        """When: salary is 'от X до Y', Then: both from and to are set."""
        html = '<h1 data-qa="vacancy-title">X</h1><span data-qa="vacancy-salary">от 120 000 до 180 000 руб.</span>'
        result = parse_modern_vacancy(html, URL)
        assert result["salary"]["from"] == 120_000
        assert result["salary"]["to"] == 180_000

    def test_no_salary_element(self) -> None:
        """When: no salary element exists, Then: all salary fields are None."""
        html = '<html><body><h1 data-qa="vacancy-title">X</h1></body></html>'
        result = parse_modern_vacancy(html, URL)
        assert result["salary"] == {"from": None, "to": None, "currency": None}


# ---------------------------------------------------------------------------
# Date parsing tests
# ---------------------------------------------------------------------------

class TestDateParsing:
    """Given: different date formats from hh.ru HTML."""

    def test_russian_date_legacy(self) -> None:
        """When: date is '12 января 2011', Then: output is '2011-01-12'."""
        html = '<span class="vacancy-creation-date">12 января 2011</span>'
        result = parse_legacy_vacancy(html, URL)
        assert result["published_at"] == "2011-01-12"

    def test_dot_format_modern(self) -> None:
        """When: date is '15.06.2023', Then: output is '2023-06-15'."""
        html = '<h1 data-qa="vacancy-title">X</h1><p class="vacancy-creation-time">15.06.2023</p>'
        result = parse_modern_vacancy(html, URL)
        assert result["published_at"] == "2023-06-15"

    def test_iso_passthrough(self) -> None:
        """When: date is already ISO, Then: returned unchanged."""
        html = '<h1 data-qa="vacancy-title">X</h1><p class="vacancy-creation-time">2024-01-31</p>'
        result = parse_modern_vacancy(html, URL)
        assert result["published_at"] == "2024-01-31"

    def test_missing_date_returns_none(self) -> None:
        """When: no date element exists, Then: published_at is None."""
        html = '<html><body><h1 data-qa="vacancy-title">X</h1></body></html>'
        result = parse_modern_vacancy(html, URL)
        assert result["published_at"] is None
