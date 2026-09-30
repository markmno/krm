"""Tests for Trudvsem.ru open data API collector (phase0/collectors/trudvsem.py).

All HTTP calls are mocked — no real API calls are made.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from krm.phase0.collectors.trudvsem import TrudvsemCollector
from krm.phase0.schema import map_trudvsem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trudvsem_vacancy(
    *,
    vacancy_id: str = "123",
    name: str = "Научный сотрудник",
    company: str = "НИИ РАН",
    region: str = "Москва",
    salary_min: int | None = 50000,
    salary_max: int | None = 80000,
    source: str = "hh.ru",
    creation_date: str = "2020-05-15",
    url: str = "https://trudvsem.ru/v/123",
) -> dict:
    """Build a Trudvsem API vacancy dict matching the real response shape."""
    vacancy: dict = {
        "id": vacancy_id,
        "vacancy_name": name,
        "company": {"name": company},
        "region": {"name": region},
        "source": source,
        "creationDate": creation_date,
        "url": url,
    }
    if salary_min is not None:
        vacancy["salary_min"] = salary_min
    if salary_max is not None:
        vacancy["salary_max"] = salary_max
    return vacancy


def _make_trudvsem_response(
    vacancies: list[dict],
    total: int | None = None,
) -> dict:
    """Build a Trudvsem API JSON response."""
    if total is None:
        total = len(vacancies)
    return {
        "results": {"vacancies": vacancies},
        "meta": {"total": total},
    }


def _mock_httpx_get(
    responses: list[dict],
    *,
    status_codes: list[int] | None = None,
) -> MagicMock:
    """Create a mock httpx.Client.get that returns the given responses in sequence.

    Each ``responses`` entry is the JSON dict returned by ``.json()``.
    """
    mock_resps: list[MagicMock] = []
    for i, body in enumerate(responses):
        mr = MagicMock()
        mr.json.return_value = body
        mr.status_code = status_codes[i] if status_codes else 200
        mr.raise_for_status = MagicMock()
        mock_resps.append(mr)

    mock_get = MagicMock()
    mock_get.side_effect = mock_resps
    return mock_get


# ---------------------------------------------------------------------------
# collect — single-page
# ---------------------------------------------------------------------------


class TestCollectSinglePage:
    """collect() with a single page of results."""

    def test_returns_mapped_vacancies(self) -> None:
        """Given a page with one vacancy, expect one mapped dict returned."""
        raw = _make_trudvsem_vacancy(vacancy_id="v1", name="Химик")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert len(result) == 1
        assert result[0]["name"] == "Химик"
        assert result[0]["id"] == "trudvsem-v1"
        assert result[0]["employer"] == {"name": "НИИ РАН"}
        assert result[0]["area"] == {"name": "Москва"}

    def test_filters_hhru_source_keeps_them(self) -> None:
        """Vacancies with source='hh.ru' are kept."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert len(result) == 1

    def test_filters_non_hhru_source_dropped(self) -> None:
        """Vacancies with source='trudvsem' (no hh.ru) are dropped."""
        raw = _make_trudvsem_vacancy(source="trudvsem")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert len(result) == 0

    def test_filters_case_insensitive(self) -> None:
        """source='HH.RU' (uppercase) is still matched."""
        raw = _make_trudvsem_vacancy(source="HH.RU")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert len(result) == 1

    def test_filters_mixed_sources(self) -> None:
        """Page with hh.ru and non-hh.ru: only hh.ru ones returned."""
        raw1 = _make_trudvsem_vacancy(vacancy_id="a", source="hh.ru")
        raw2 = _make_trudvsem_vacancy(vacancy_id="b", source="trudvsem")
        raw3 = _make_trudvsem_vacancy(vacancy_id="c", source="other")
        response_body = _make_trudvsem_response([raw1, raw2, raw3])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert len(result) == 1
        assert result[0]["id"] == "trudvsem-a"

    def test_handles_missing_source_field(self) -> None:
        """Vacancy without a source field is dropped gracefully."""
        raw = _make_trudvsem_vacancy(vacancy_id="x")
        del raw["source"]
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert len(result) == 0

    def test_handles_none_source_field(self) -> None:
        """Vacancy with source=None is treated as missing and dropped."""
        raw = _make_trudvsem_vacancy(vacancy_id="x")
        raw["source"] = None  # type: ignore[typeddict-item]
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert len(result) == 0

    def test_passes_offset_and_limit(self) -> None:
        """collect() forwards offset and limit as query params."""
        raw = _make_trudvsem_vacancy()
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            collector.collect("2020-01-01", "2020-12-31", offset=50, limit=25)

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["offset"] == 50
        assert params["limit"] == 25

    def test_adds_phase0_metadata(self) -> None:
        """Mapped result includes _phase0_source='trudvsem'."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["_phase0_source"] == "trudvsem"
        assert result[0]["_phase0_capture_ts"]  # non-empty timestamp
        assert result[0]["_phase0_original_url"] == "https://trudvsem.ru/v/123"


# ---------------------------------------------------------------------------
# collect_all — pagination
# ---------------------------------------------------------------------------


class TestCollectAll:
    """collect_all() paginates through all result pages."""

    def test_single_page(self) -> None:
        """One page with total=count: single request, returns all."""
        raw = _make_trudvsem_vacancy(vacancy_id="a", source="hh.ru")
        response_body = _make_trudvsem_response([raw], total=1)
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect_all("2020-01-01", "2020-12-31")

        assert len(result) == 1
        assert result[0]["id"] == "trudvsem-a"

    def test_two_pages(self) -> None:
        """total=150 with limit=100: first page returns 100, second returns 50."""
        # Page 1: total=150, 100 items (with hh.ru source)
        p1_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"v{i:03d}", source="hh.ru")
            for i in range(100)
        ]
        p1_response = _make_trudvsem_response(p1_vacancies, total=150)

        # Page 2: offset=100, 50 items
        p2_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"v{i:03d}", source="hh.ru")
            for i in range(100, 150)
        ]
        p2_response = _make_trudvsem_response(p2_vacancies, total=150)

        mock_get = _mock_httpx_get([p1_response, p2_response])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect_all("2020-01-01", "2020-12-31")

        assert mock_get.call_count == 2
        assert len(result) == 150

    def test_three_pages(self) -> None:
        """total=250 with limit=100: three pages (100 + 100 + 50)."""
        p1_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"p1-{i:03d}", source="hh.ru")
            for i in range(100)
        ]
        p1 = _make_trudvsem_response(p1_vacancies, total=250)

        p2_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"p2-{i:03d}", source="hh.ru")
            for i in range(100)
        ]
        p2 = _make_trudvsem_response(p2_vacancies, total=250)

        p3_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"p3-{i:03d}", source="hh.ru")
            for i in range(50)
        ]
        p3 = _make_trudvsem_response(p3_vacancies, total=250)

        mock_get = _mock_httpx_get([p1, p2, p3])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect_all("2020-01-01", "2020-12-31")

        assert mock_get.call_count == 3
        assert len(result) == 250

    def test_filtering_across_pages(self) -> None:
        """Page 1 has 100 hh.ru items, page 2 has 50 non-hh.ru only."""
        p1_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"hh-{i:03d}", source="hh.ru")
            for i in range(100)
        ]
        p1 = _make_trudvsem_response(p1_vacancies, total=150)

        p2_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"other-{i:03d}", source="trudvsem")
            for i in range(50)
        ]
        p2 = _make_trudvsem_response(p2_vacancies, total=150)

        mock_get = _mock_httpx_get([p1, p2])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect_all("2020-01-01", "2020-12-31")

        # Only page 1 hh.ru items kept; page 2 trudvsem items filtered out
        assert len(result) == 100
        for r in result:
            assert r["id"].startswith("trudvsem-hh-")

    def test_pagination_offset_increments(self) -> None:
        """Second page request uses offset=100 (default limit) after first page."""
        # total=200 triggers 2 pages with limit=100
        p1_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"a{i:03d}", source="hh.ru")
            for i in range(100)
        ]
        p1 = _make_trudvsem_response(p1_vacancies, total=200)

        p2_vacancies = [
            _make_trudvsem_vacancy(vacancy_id=f"b{i:03d}", source="hh.ru")
            for i in range(100)
        ]
        p2 = _make_trudvsem_response(p2_vacancies, total=200)

        mock_get = _mock_httpx_get([p1, p2])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            collector.collect_all("2020-01-01", "2020-12-31")

        # First call: offset=0
        assert mock_get.call_args_list[0][1]["params"]["offset"] == 0
        # Second call: offset=100 (default limit)
        assert mock_get.call_args_list[1][1]["params"]["offset"] == 100


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """_rate_limit() enforces the configured requests-per-second cap."""

    def test_first_request_no_delay(self) -> None:
        """First request should not sleep — elapsed time is huge."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        # Simulate real monotonic clock: 100000 seconds since boot.
        # _last_request_time starts at 0.0, so elapsed ≈ 100000 >> 1.0.
        with patch("time.monotonic") as mock_mono, patch("time.sleep") as mock_sleep:
            mock_mono.side_effect = [100000.0, 100000.0]
            with patch.object(httpx.Client, "get", mock_get):
                collector = TrudvsemCollector(rate_limit_rps=1)
                collector.collect("2020-01-01", "2020-12-31")

        mock_sleep.assert_not_called()

    def test_second_request_too_fast_waits(self) -> None:
        """Second request within <1s of the first sleeps for the remainder."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body, response_body])

        # Simulate: first call at T=100000.0, second at T=100000.2
        # elapsed = 0.2, needs 0.8 more sleep for rate_limit_rps=1
        with patch("time.monotonic") as mock_mono, patch("time.sleep") as mock_sleep:
            mock_mono.side_effect = [
                100000.0,  # _rate_limit check (first call)
                100000.0,  # post-request: set last_request_time
                100000.2,  # _rate_limit check (second call): elapsed = 0.2
                100001.0,  # post-request: set last_request_time
            ]
            with patch.object(httpx.Client, "get", mock_get):
                collector = TrudvsemCollector(rate_limit_rps=1)
                collector.collect("2020-01-01", "2020-12-31")
                collector.collect("2020-01-01", "2020-12-31")

        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] == pytest.approx(0.8)

    def test_rate_limit_zero_skips_sleep(self) -> None:
        """rate_limit_rps=0 means no rate limiting."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body, response_body])

        with patch("time.sleep") as mock_sleep:
            with patch.object(httpx.Client, "get", mock_get):
                collector = TrudvsemCollector(rate_limit_rps=0)
                collector.collect("2020-01-01", "2020-12-31")
                collector.collect("2020-01-01", "2020-12-31")

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Empty response
# ---------------------------------------------------------------------------


class TestEmptyResponse:
    """Edge case: API returns zero vacancies."""

    def test_empty_vacancies_list(self) -> None:
        """results.vacancies is an empty list."""
        response_body = {"results": {"vacancies": []}, "meta": {"total": 0}}
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result == []

    def test_collect_all_empty(self) -> None:
        """collect_all with total=0 returns empty list."""
        response_body = {"results": {"vacancies": []}, "meta": {"total": 0}}
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect_all("2020-01-01", "2020-12-31")

        assert result == []
        # Only one request — loop never enters because offset (0) >= total (0) → wait, no
        # Actually: offset=0, total=0, first request paged, offset becomes 100.
        # Then while 100 < 0 is False, so only one request.
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# Schema mapping: verify output has hh.ru API field names
# ---------------------------------------------------------------------------


class TestSchemaMappingOutput:
    """Mapped output uses hh.ru API field names, not Trudvsem Russian names."""

    def test_uses_hhru_field_names(self) -> None:
        """Result keys match the hh.ru schema: 'name', 'employer', 'area', 'salary'."""
        raw = _make_trudvsem_vacancy(
            vacancy_id="v1",
            name="Физик-теоретик",
            company="ИТЭФ",
            region="Москва",
            salary_min=100_000,
            salary_max=150_000,
            source="hh.ru",
        )
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        mapped = result[0]

        # Russian Trudvsem field names must NOT appear in output
        assert "vacancy_name" not in mapped
        assert "vacancy_description" not in mapped
        assert "company" not in mapped
        assert "region" not in mapped
        assert "salary_min" not in mapped
        assert "salary_max" not in mapped

        # hh.ru schema field names must be present
        assert mapped["name"] == "Физик-теоретик"
        assert mapped["employer"] == {"name": "ИТЭФ"}
        assert mapped["area"] == {"name": "Москва"}
        assert mapped["salary"] == {
            "from": 100_000,
            "to": 150_000,
            "currency": "RUR",
        }

    def test_description_mapped(self) -> None:
        """vacancy_description is mapped to 'description'."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        raw["vacancy_description"] = "Проведение экспериментов"
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["description"] == "Проведение экспериментов"
        assert "vacancy_description" not in result[0]

    def test_id_prefixed_with_trudvsem(self) -> None:
        """Trudvsem raw ID gets 'trudvsem-' prefix in mapped output."""
        raw = _make_trudvsem_vacancy(vacancy_id="abc-999", source="hh.ru")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["id"] == "trudvsem-abc-999"

    def test_key_skills_preserved(self) -> None:
        """key_skills field is converted to canonical format."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        raw["key_skills"] = [{"name": "Python"}, {"name": "математика"}]
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["key_skills"] == [
            {"name": "Python"},
            {"name": "математика"},
        ]

    def test_capture_ts_unique_per_collect_call(self) -> None:
        """Each collect() call generates a unique capture_ts."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body, response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector(rate_limit_rps=0)
            result1 = collector.collect("2020-01-01", "2020-12-31")
            result2 = collector.collect("2020-01-01", "2020-12-31")

        ts1 = result1[0]["_phase0_capture_ts"]
        ts2 = result2[0]["_phase0_capture_ts"]
        assert ts1
        assert ts2
        assert ts1 != ts2


# ---------------------------------------------------------------------------
# Missing fields — partial API data
# ---------------------------------------------------------------------------


class TestMissingFields:
    """API returns partial data; mapper fills None for missing optional fields."""

    def test_minimal_vacancy_only_name_and_id(self) -> None:
        """Only id and vacancy_name present in API response."""
        raw: dict = {
            "id": "min1",
            "vacancy_name": "Минимальная вакансия",
            "source": "hh.ru",
        }
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        mapped = result[0]
        assert mapped["name"] == "Минимальная вакансия"
        assert mapped["id"] == "trudvsem-min1"
        # Optional fields that were not in the raw response
        assert "description" not in mapped or mapped.get("description") is None
        assert "employer" not in mapped or mapped.get("employer") is None
        assert "area" not in mapped or mapped.get("area") is None
        assert "salary" not in mapped or mapped.get("salary") is None

    def test_no_company_no_employer_in_output(self) -> None:
        """Without company/company_name, employer is absent from output."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        del raw["company"]
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert "employer" not in result[0]

    def test_no_region_no_area_in_output(self) -> None:
        """Without region/region_name, area is absent from output."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        del raw["region"]
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert "area" not in result[0]

    def test_salary_only_min(self) -> None:
        """When only salary_min is present, salary.to is None."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        raw["salary_min"] = 70000
        del raw["salary_max"]
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["salary"] == {
            "from": 70000,
            "to": None,
            "currency": "RUR",
        }

    def test_salary_only_max(self) -> None:
        """When only salary_max is present, salary.from is None."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        del raw["salary_min"]
        raw["salary_max"] = 120000
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["salary"] == {
            "from": None,
            "to": 120000,
            "currency": "RUR",
        }

    def test_company_name_string_fallback(self) -> None:
        """company_name string (not dict) is mapped to employer."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        del raw["company"]
        raw["company_name"] = "Росатом"
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["employer"] == {"name": "Росатом"}

    def test_region_name_string_fallback(self) -> None:
        """region_name string (not dict) is mapped to area."""
        raw = _make_trudvsem_vacancy(source="hh.ru")
        del raw["region"]
        raw["region_name"] = "Казань"
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2020-01-01", "2020-12-31")

        assert result[0]["area"] == {"name": "Казань"}


# ---------------------------------------------------------------------------
# Integration: collector + mapper round-trip
# ---------------------------------------------------------------------------


class TestCollectorMapperRoundTrip:
    """Collector output passes through schema mapper correctly."""

    def test_full_vacancy_no_keys_dropped(self) -> None:
        """Full vacancy with all fields survives collect → mapper round-trip."""
        raw: dict = {
            "id": "full1",
            "vacancy_name": "Научный сотрудник",
            "vacancy_description": "Исследования в области физики",
            "company": {"name": "НИИ РАН"},
            "region": {"name": "Москва"},
            "salary_min": 100000,
            "salary_max": 200000,
            "salary_currency": "RUR",
            "source": "hh.ru",
            "creationDate": "2024-06-01",
            "url": "https://trudvsem.ru/v/full1",
            "key_skills": [{"name": "Python"}, {"name": "Статистика"}],
        }
        response_body = _make_trudvsem_response([raw])
        mock_get = _mock_httpx_get([response_body])

        with patch.object(httpx.Client, "get", mock_get):
            collector = TrudvsemCollector()
            result = collector.collect("2024-01-01", "2024-12-31")

        mapped = result[0]
        assert mapped["id"] == "trudvsem-full1"
        assert mapped["name"] == "Научный сотрудник"
        assert mapped["description"] == "Исследования в области физики"
        assert mapped["employer"] == {"name": "НИИ РАН"}
        assert mapped["area"] == {"name": "Москва"}
        assert mapped["salary"] == {"from": 100000, "to": 200000, "currency": "RUR"}
        assert mapped["_phase0_source"] == "trudvsem"
        assert mapped["_phase0_original_url"] == "https://trudvsem.ru/v/full1"
        assert mapped["key_skills"] == [{"name": "Python"}, {"name": "Статистика"}]
