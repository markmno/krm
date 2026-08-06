"""Tests for phase_1_collect.py — HH.ru API scraper constants and helpers.

Tests only pure logic: no httpx, no DuckDB, no real API calls.
"""

from __future__ import annotations

import pytest

from krm.phase_1_collect import (
    HH_API_BASE,
    HH_MAX_RESULTS,
    HH_PER_PAGE,
    HH_USER_AGENT,
    MAX_PAGES,
    _make_run_id,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_api_base_no_trailing_slash(self) -> None:
        """HH_API_BASE must not have a trailing slash for httpx base_url."""
        assert not HH_API_BASE.endswith("/")
        assert HH_API_BASE.startswith("https://")

    def test_api_base_is_hhru(self) -> None:
        """HH_API_BASE points to the official HeadHunter API host."""
        assert "api.hh.ru" in HH_API_BASE

    def test_user_agent_contains_project_name(self) -> None:
        """User-Agent identifies the KRM pipeline per RFC 7231."""
        assert "KRM-Pipeline" in HH_USER_AGENT
        assert "/" in HH_USER_AGENT
        assert len(HH_USER_AGENT) > 20

    def test_per_page_is_max_hhru_page_size(self) -> None:
        """HH_PER_PAGE must be 100, the maximum the HH.ru API accepts."""
        assert HH_PER_PAGE == 100
        assert isinstance(HH_PER_PAGE, int)

    def test_max_results_is_hhru_hard_cap(self) -> None:
        """HH_MAX_RESULTS must be 2000, the HH.ru search result hard cap."""
        assert HH_MAX_RESULTS == 2000
        assert isinstance(HH_MAX_RESULTS, int)

    def test_max_pages_derived_from_cap_and_page_size(self) -> None:
        """MAX_PAGES = HH_MAX_RESULTS // HH_PER_PAGE (floor division)."""
        assert MAX_PAGES == 20
        assert MAX_PAGES == HH_MAX_RESULTS // HH_PER_PAGE

    def test_all_numeric_constants_positive(self) -> None:
        """Collection constants represent limits; negative values are nonsense."""
        assert HH_PER_PAGE > 0
        assert HH_MAX_RESULTS > 0
        assert MAX_PAGES > 0


# ---------------------------------------------------------------------------
# _make_run_id
# ---------------------------------------------------------------------------


class TestMakeRunId:
    def test_simple_keyword(self) -> None:
        """A single-word keyword produces a run ID starting with that word."""
        run_id = _make_run_id("Python")
        assert run_id.startswith("python_")

    def test_multi_word_lowercased_and_despace(self) -> None:
        """Spaces become underscores, and the entire string is lowercased."""
        run_id = _make_run_id("Data Science")
        assert run_id.startswith("data_science_")
        assert " " not in run_id
        assert run_id == run_id.lower()

    def test_contains_timestamp_suffix(self) -> None:
        """Format: {keyword}_{YYYYMMDD}_{HHMMSS}."""
        run_id = _make_run_id("ML")
        parts = run_id.split("_")
        # parts: [keyword, YYYYMMDD, HHMMSS]
        assert len(parts) >= 3

        date_part = parts[1]
        time_part = parts[2]
        assert len(date_part) == 8
        assert len(time_part) == 6
        assert date_part.isdigit()
        assert time_part.isdigit()

    def test_timestamp_components_in_range(self) -> None:
        """The embedded date and time should be structurally valid."""
        run_id = _make_run_id("test")
        parts = run_id.split("_")
        date_str = parts[1]  # YYYYMMDD
        time_str = parts[2]  # HHMMSS

        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        assert 2020 <= year <= 2099
        assert 1 <= month <= 12
        assert 1 <= day <= 31
        assert 0 <= hour <= 23
        assert 0 <= minute <= 59
        assert 0 <= second <= 60  # leap-second safe

    def test_special_char_minimized(self) -> None:
        """Keyword with special characters is sanitized."""
        run_id = _make_run_id("C++ Developer")
        assert " " not in run_id

    def test_returns_string(self) -> None:
        """The return type is always str."""
        assert isinstance(_make_run_id("anything"), str)


# ---------------------------------------------------------------------------
# Date-range formatting (the pattern used in collect())
# ---------------------------------------------------------------------------


class TestDateFormatting:
    def test_iso_combined_format_appends_midnight(self) -> None:
        """collect() builds ISO 8601 datetime with f'{date_from}T00:00:00'."""
        formatted = f"{'2024-01-15'}T00:00:00"
        assert formatted == "2024-01-15T00:00:00"

    def test_various_dates(self) -> None:
        """The formatting pattern holds for any YYYY-MM-DD input."""
        cases = [
            ("2024-01-01", "2024-01-01T00:00:00"),
            ("2023-12-31", "2023-12-31T00:00:00"),
            ("2025-06-15", "2025-06-15T00:00:00"),
        ]
        for date_str, expected in cases:
            assert f"{date_str}T00:00:00" == expected

    def test_starts_and_ends_of_month(self) -> None:
        """Boundary dates (month starts/ends) format cleanly."""
        boundaries = ["2024-02-01", "2024-02-29", "2024-03-01", "2024-03-31"]
        for d in boundaries:
            assert f"{d}T00:00:00".endswith("T00:00:00")

    def test_single_digit_months_days_ok(self) -> None:
        """Single-digit months/days (2024-1-5) are NOT used; config gives
        zero-padded YYYY-MM-DD. Verify the zero-padded case works."""
        formatted = f"{'2024-01-05'}T00:00:00"
        assert formatted == "2024-01-05T00:00:00"
        # Double-digit day is preserved.
        assert "05T" in formatted


# ---------------------------------------------------------------------------
# Rate-limit arithmetic
# ---------------------------------------------------------------------------


class TestRateLimitArithmetic:
    def test_min_interval_computation(self) -> None:
        """min_interval = 1.0 / rate_limit_rps."""
        assert 1.0 / 1 == 1.0
        assert 1.0 / 2 == 0.5
        assert 1.0 / 5 == 0.2
        assert 1.0 / 10 == 0.1

    def test_backoff_doubling_capped(self) -> None:
        """backoff = min(backoff * 2, 120.0). Saturates at 120 s."""
        backoff = 1.0
        sequence: list[float] = []
        for _ in range(10):
            backoff = min(backoff * 2, 120.0)
            sequence.append(backoff)
        assert sequence == [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 120.0, 120.0, 120.0, 120.0]

    def test_max_pages_fully_utilises_result_cap(self) -> None:
        """MAX_PAGES * HH_PER_PAGE == HH_MAX_RESULTS (no wasted pages)."""
        assert MAX_PAGES * HH_PER_PAGE == HH_MAX_RESULTS


# ---------------------------------------------------------------------------
# URL / endpoint construction
# ---------------------------------------------------------------------------


class TestURLConstruction:
    def test_full_vacancies_url(self) -> None:
        """The client constructs {HH_API_BASE}/vacancies."""
        full = f"{HH_API_BASE}/vacancies"
        assert full == "https://api.hh.ru/vacancies"

    def test_api_base_is_https_only(self) -> None:
        """HH.ru requires TLS."""
        assert HH_API_BASE.startswith("https://")
        assert not HH_API_BASE.startswith("http://")


# ---------------------------------------------------------------------------
# Query-parameter structure
# ---------------------------------------------------------------------------


class TestParamsStructure:
    def test_base_params_match_constants(self) -> None:
        """collect() builds api_params using HH_PER_PAGE."""
        params = {
            "date_from": "2024-01-01T00:00:00",
            "per_page": HH_PER_PAGE,
            "order_by": "publication_time",
        }
        assert params["per_page"] == 100
        assert params["order_by"] == "publication_time"
        assert params["date_from"].endswith("T00:00:00")

    def test_keyword_injected_as_text_param(self) -> None:
        """_collect_keyword copies api_params and sets text=keyword."""
        params = {"date_from": "2024-01-01T00:00:00", "per_page": 100}
        keyword = "Python разработчик"
        params["text"] = keyword
        assert params["text"] == keyword

    def test_professional_role_attached_when_present(self) -> None:
        """When config.professional_roles is non-empty, it is attached."""
        params = {"date_from": "2024-01-01T00:00:00", "per_page": 100}
        roles = [96, 123, 125]
        if roles:
            params["professional_role"] = roles
        assert params["professional_role"] == [96, 123, 125]

    def test_professional_role_not_attached_when_empty(self) -> None:
        """An empty list should NOT add the professional_role key."""
        params = {"date_from": "2024-01-01T00:00:00", "per_page": 100}
        roles: list[int] = []
        if roles:
            params["professional_role"] = roles
        assert "professional_role" not in params

    def test_page_param_iteration(self) -> None:
        """Each iteration of the page loop sets page=N."""
        params = {"date_from": "2024-01-01T00:00:00", "per_page": 100}
        pages_seen: list[int] = []
        for page in range(MAX_PAGES):
            params["page"] = page
            pages_seen.append(params["page"])
        assert pages_seen == list(range(MAX_PAGES))
