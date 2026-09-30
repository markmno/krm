"""Tests for phase0/schema.py — schema mapper and normaliser."""

from __future__ import annotations

import pytest

from krm.phase0.schema import (
    OPTIONAL_FIELDS,
    PHASE0_META_FIELDS,
    REQUIRED_FIELDS,
    SOURCE_TYPES,
    map_hhru_wayback,
    map_linkedin_wayback,
    map_rostud,
    map_trudvsem,
    normalize_record,
    validate_record,
)


# ======================================================================
# Constants
# ======================================================================


class TestConstants:
    def test_required_fields_are_a_set(self) -> None:
        assert isinstance(REQUIRED_FIELDS, set)
        assert "id" in REQUIRED_FIELDS
        assert "name" in REQUIRED_FIELDS

    def test_optional_fields_are_a_set(self) -> None:
        assert isinstance(OPTIONAL_FIELDS, set)
        assert len(OPTIONAL_FIELDS) > 0

    def test_required_and_optional_are_disjoint(self) -> None:
        assert REQUIRED_FIELDS.isdisjoint(OPTIONAL_FIELDS)

    def test_meta_fields_have_phase0_prefix(self) -> None:
        for field in PHASE0_META_FIELDS:
            assert field.startswith("_phase0_")

    def test_source_types_include_all_four(self) -> None:
        assert set(SOURCE_TYPES) >= {
            "wayback-hhru",
            "wayback-linkedin",
            "trudvsem",
            "rostud",
        }, f"Expected at least 4 base source types, got {SOURCE_TYPES}"


# ======================================================================
# map_hhru_wayback
# ======================================================================


class TestMapHHruWayback:
    """Direct mapping: hh.ru extracted fields already use API field names."""

    def test_passthrough_id_and_name(self) -> None:
        raw = {"id": "12345", "name": "Python Developer"}
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/12345", "20240101000000")

        assert result["id"] == "12345"
        assert result["name"] == "Python Developer"

    def test_extracts_id_from_url_when_missing(self) -> None:
        raw = {"name": "Data Scientist"}
        result = map_hhru_wayback(
            raw, "https://hh.ru/vacancy/99999?query=ds", "20240101000000"
        )

        assert result["id"] == "99999"

    def test_adds_phase0_metadata(self) -> None:
        raw = {"id": "1", "name": "X"}
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/1", "2024-12-01T00:00:00")

        assert result["_phase0_source"] == "wayback-hhru"
        assert result["_phase0_capture_ts"] == "2024-12-01T00:00:00"
        assert result["_phase0_original_url"] == "https://hh.ru/vacancy/1"

    def test_passthrough_optional_fields(self) -> None:
        raw = {
            "id": "42",
            "name": "Engineer",
            "description": "Build things",
            "employer": {"name": "Acme Corp"},
            "area": {"name": "Moscow"},
            "salary": {"from": 100_000, "to": 200_000, "currency": "RUR"},
        }
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/42", "ts")

        assert result["description"] == "Build things"
        assert result["employer"] == {"name": "Acme Corp"}
        assert result["area"] == {"name": "Moscow"}
        assert result["salary"] == {"from": 100_000, "to": 200_000, "currency": "RUR"}

    def test_coerces_key_skills_strings_to_dicts(self) -> None:
        raw = {"id": "1", "name": "X", "key_skills": ["Python", "SQL"]}
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/1", "ts")

        assert result["key_skills"] == [
            {"name": "Python"},
            {"name": "SQL"},
        ]

    def test_key_skills_already_dicts_preserved(self) -> None:
        raw = {
            "id": "1",
            "name": "X",
            "key_skills": [{"name": "Python"}, {"name": "SQL"}],
        }
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/1", "ts")

        assert result["key_skills"] == [
            {"name": "Python"},
            {"name": "SQL"},
        ]

    def test_empty_key_skills_becomes_empty_list(self) -> None:
        raw = {"id": "1", "name": "X", "key_skills": []}
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/1", "ts")

        assert result["key_skills"] == []

    def test_missing_key_skills_becomes_empty_list(self) -> None:
        raw = {"id": "1", "name": "X"}
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/1", "ts")

        assert result["key_skills"] == []

    def test_coerces_professional_roles(self) -> None:
        raw = {
            "id": "1",
            "name": "X",
            "professional_roles": [
                {"id": 96, "name": "Программист, разработчик"}
            ],
        }
        result = map_hhru_wayback(raw, "https://hh.ru/vacancy/1", "ts")

        assert result["professional_roles"] == [
            {"id": 96, "name": "Программист, разработчик"},
        ]


# ======================================================================
# map_linkedin_wayback
# ======================================================================


class TestMapLinkedInWayback:
    """LinkedIn-specific field names mapped to hh.ru schema."""

    def test_maps_employer_dict_to_employer_name(self) -> None:
        raw = {
            "name": "Software Engineer",
            "employer": {"name": "Google"},
        }
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert result["employer"] == {"name": "Google"}

    def test_maps_employer_string(self) -> None:
        raw = {"name": "SE", "employer": "Acme"}
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert result["employer"] == {"name": "Acme"}

    def test_maps_area_dict_to_area_name(self) -> None:
        raw = {
            "name": "SE",
            "area": {"name": "San Francisco"},
        }
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert result["area"] == {"name": "San Francisco"}

    def test_name_direct_passthrough(self) -> None:
        raw = {"name": "Senior Backend Engineer"}
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert result["name"] == "Senior Backend Engineer"

    def test_description_passthrough(self) -> None:
        raw = {"name": "SE", "description": "Build APIs"}
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert result["description"] == "Build APIs"

    def test_industry_prepended_to_description(self) -> None:
        raw = {
            "name": "SE",
            "description": "Build APIs",
            "industry": "Technology",
        }
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert result["description"] == "[Industry: Technology] Build APIs"

    def test_industry_no_description_stays_empty(self) -> None:
        raw = {"name": "SE", "industry": "Finance"}
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert "description" not in result

    def test_extracts_id_from_jobs2_view_url(self) -> None:
        raw = {"name": "SE"}
        result = map_linkedin_wayback(
            raw, "https://www.linkedin.com/jobs2/view/12345", "ts"
        )

        assert result["id"] == "linkedin-12345"

    def test_extracts_id_from_slug_url(self) -> None:
        raw = {"name": "SE"}
        result = map_linkedin_wayback(
            raw,
            "https://www.linkedin.com/jobs/view/software-engineer-1234567890",
            "ts",
        )

        assert result["id"] == "linkedin-1234567890"

    def test_uses_raw_id_when_present(self) -> None:
        raw = {"id": "98765", "name": "SE"}
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/12345", "ts"
        )

        assert result["id"] == "linkedin-98765"

    def test_adds_phase0_metadata(self) -> None:
        raw = {"name": "SE"}
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/42", "2024-06-01"
        )

        assert result["_phase0_source"] == "wayback-linkedin"
        assert result["_phase0_capture_ts"] == "2024-06-01"
        assert result["_phase0_original_url"] == "https://linkedin.com/jobs/view/42"

    def test_key_skills_coerced(self) -> None:
        raw = {"name": "SE", "key_skills": ["Java", "Spring"]}
        result = map_linkedin_wayback(
            raw, "https://linkedin.com/jobs/view/1", "ts"
        )

        assert result["key_skills"] == [
            {"name": "Java"},
            {"name": "Spring"},
        ]


# ======================================================================
# map_trudvsem
# ======================================================================


class TestMapTrudvsem:
    """Russian field names from Trudvsem API mapped to hh.ru schema."""

    def test_maps_vacancy_name_to_name(self) -> None:
        raw = {"id": "v1", "vacancy_name": "Инженер-программист"}
        result = map_trudvsem(raw, "ts")

        assert result["name"] == "Инженер-программист"

    def test_maps_vacancy_description_to_description(self) -> None:
        raw = {
            "id": "v1",
            "vacancy_name": "Инженер",
            "vacancy_description": "Разработка ПО",
        }
        result = map_trudvsem(raw, "ts")

        assert result["description"] == "Разработка ПО"

    def test_maps_company_dict_to_employer(self) -> None:
        raw = {
            "id": "v1",
            "vacancy_name": "Инженер",
            "company": {"name": "Яндекс", "inn": "7700000000"},
        }
        result = map_trudvsem(raw, "ts")

        assert result["employer"] == {"name": "Яндекс"}

    def test_maps_company_name_string_to_employer(self) -> None:
        raw = {
            "id": "v1",
            "vacancy_name": "Инженер",
            "company_name": "Сбер",
        }
        result = map_trudvsem(raw, "ts")

        assert result["employer"] == {"name": "Сбер"}

    def test_maps_region_dict_to_area(self) -> None:
        raw = {
            "id": "v1",
            "vacancy_name": "Инженер",
            "region": {"name": "Москва", "code": "77"},
        }
        result = map_trudvsem(raw, "ts")

        assert result["area"] == {"name": "Москва"}

    def test_maps_region_name_string_to_area(self) -> None:
        raw = {
            "id": "v1",
            "vacancy_name": "Инженер",
            "region_name": "Казань",
        }
        result = map_trudvsem(raw, "ts")

        assert result["area"] == {"name": "Казань"}

    def test_maps_salary_min_max(self) -> None:
        raw = {
            "id": "v1",
            "vacancy_name": "Инженер",
            "salary_min": 80_000,
            "salary_max": 120_000,
            "salary_currency": "RUR",
        }
        result = map_trudvsem(raw, "ts")

        assert result["salary"] == {
            "from": 80_000,
            "to": 120_000,
            "currency": "RUR",
        }

    def test_salary_defaults_currency(self) -> None:
        raw = {
            "id": "v1",
            "vacancy_name": "Инженер",
            "salary_min": 50_000,
        }
        result = map_trudvsem(raw, "ts")

        assert result["salary"] == {
            "from": 50_000,
            "to": None,
            "currency": "RUR",
        }

    def test_no_salary_when_no_fields(self) -> None:
        raw = {"id": "v1", "vacancy_name": "Инженер"}
        result = map_trudvsem(raw, "ts")

        assert "salary" not in result

    def test_prepends_trudvsem_to_id(self) -> None:
        raw = {"id": "abc123", "vacancy_name": "Инженер"}
        result = map_trudvsem(raw, "ts")

        assert result["id"] == "trudvsem-abc123"

    def test_uses_vacancy_id_fallback(self) -> None:
        raw = {"vacancy_id": "v42", "vacancy_name": "Инженер"}
        result = map_trudvsem(raw, "ts")

        assert result["id"] == "trudvsem-v42"

    def test_adds_phase0_metadata(self) -> None:
        raw = {"id": "x", "vacancy_name": "Инженер", "url": "https://trudvsem.ru/vacancy/1"}
        result = map_trudvsem(raw, "2025-01-01T12:00:00")

        assert result["_phase0_source"] == "trudvsem"
        assert result["_phase0_capture_ts"] == "2025-01-01T12:00:00"
        assert result["_phase0_original_url"] == "https://trudvsem.ru/vacancy/1"


# ======================================================================
# map_rostud
# ======================================================================


class TestMapRostrud:
    """Flexible Russian column-name matching for Rostrud CSV rows."""

    def test_prefers_russian_column_names(self) -> None:
        raw = {
            "id": "r1",
            "Должность": "Лаборант",
            "Работодатель": "НИИ Физики",
            "Регион": "Новосибирск",
            "Описание": "Анализ образцов",
        }
        result = map_rostud(raw, "ts")

        assert result["name"] == "Лаборант"
        assert result["employer"] == {"name": "НИИ Физики"}
        assert result["area"] == {"name": "Новосибирск"}
        assert result["description"] == "Анализ образцов"

    def test_falls_back_to_latin_names(self) -> None:
        raw = {
            "id": "r1",
            "position": "Lab Assistant",
            "employer": "Physics Institute",
            "location": "Novosibirsk",
        }
        result = map_rostud(raw, "ts")

        assert result["name"] == "Lab Assistant"
        assert result["employer"] == {"name": "Physics Institute"}
        assert result["area"] == {"name": "Novosibirsk"}

    def test_falls_back_to_generic_name_key(self) -> None:
        raw = {"id": "r1", "name": "Generic Worker"}
        result = map_rostud(raw, "ts")

        assert result["name"] == "Generic Worker"

    def test_uses_raw_id_when_present(self) -> None:
        raw = {"id": "rostud-42", "Должность": "Инженер"}
        result = map_rostud(raw, "ts")

        assert result["id"] == "rostud-42"

    def test_generates_hash_id_when_no_raw_id(self) -> None:
        raw = {"Должность": "Химик", "Работодатель": "Лаб"}
        result = map_rostud(raw, "20240101000000")

        assert result["id"]  # non-empty
        assert isinstance(result["id"], str)

    def test_generated_ids_are_stable_for_same_input(self) -> None:
        raw = {"Должность": "Химик", "Работодатель": "Лаб"}
        id1 = map_rostud(dict(raw), "ts")["id"]
        id2 = map_rostud(dict(raw), "ts")["id"]

        assert id1 == id2

    def test_adds_phase0_metadata(self) -> None:
        raw = {
            "id": "r1",
            "Должность": "Инженер",
            "source_url": "https://rostrud.gov.ru/data/2024.csv",
        }
        result = map_rostud(raw, "2024-03-15")

        assert result["_phase0_source"] == "rostud"
        assert result["_phase0_capture_ts"] == "2024-03-15"
        assert result["_phase0_original_url"] == "https://rostrud.gov.ru/data/2024.csv"

    def test_key_skills_coerced(self) -> None:
        raw = {
            "id": "r1",
            "Должность": "Инженер",
            "key_skills": ["Черчение", "AutoCAD"],
        }
        result = map_rostud(raw, "ts")

        assert result["key_skills"] == [
            {"name": "Черчение"},
            {"name": "AutoCAD"},
        ]

    def test_missing_everything_produces_empty_name(self) -> None:
        raw: dict[str, str] = {}
        result = map_rostud(raw, "ts")

        assert result["name"] == ""


# ======================================================================
# normalize_record
# ======================================================================


class TestNormalizeRecord:
    """Fill missing optionals and enforce required fields."""

    def test_fills_missing_optional_fields_with_none(self) -> None:
        mapped = {"id": "v1", "name": "Engineer"}
        result = normalize_record(mapped)

        for field in OPTIONAL_FIELDS:
            assert result.get(field) is None, f"Field {field!r} should be None"

    def test_preserves_existing_optional_fields(self) -> None:
        mapped = {
            "id": "v1",
            "name": "Engineer",
            "description": "desc",
            "salary": {"from": 1, "to": 2},
        }
        result = normalize_record(mapped)

        assert result["description"] == "desc"
        assert result["salary"] == {"from": 1, "to": 2}

    def test_fills_missing_meta_fields(self) -> None:
        mapped = {"id": "v1", "name": "Engineer"}
        result = normalize_record(mapped)

        for meta in PHASE0_META_FIELDS:
            assert result.get(meta) is None, f"Meta field {meta!r} should be filled"

    def test_raises_value_error_when_name_missing(self) -> None:
        mapped = {"id": "v1"}
        with pytest.raises(ValueError, match="name"):
            normalize_record(mapped)

    def test_raises_value_error_when_name_empty(self) -> None:
        mapped = {"id": "v1", "name": ""}
        with pytest.raises(ValueError, match="name"):
            normalize_record(mapped)

    def test_raises_value_error_when_id_missing(self) -> None:
        mapped = {"name": "Engineer"}
        with pytest.raises(ValueError, match="id"):
            normalize_record(mapped)

    def test_does_not_overwrite_non_none_optionals_with_none(self) -> None:
        mapped = {
            "id": "v1",
            "name": "Engineer",
            "area": {"name": "Moscow"},
            "employer": {"name": "Acme"},
        }
        result = normalize_record(mapped)

        assert result["area"] == {"name": "Moscow"}
        assert result["employer"] == {"name": "Acme"}


# ======================================================================
# validate_record
# ======================================================================


class TestValidateRecord:
    """Structural validation of normalised records."""

    def test_valid_record_returns_empty_errors(self) -> None:
        record = {
            "id": "v1",
            "name": "Engineer",
            "_phase0_source": "wayback-hhru",
            "_phase0_capture_ts": "2024-01-01",
            "_phase0_original_url": "https://hh.ru/vacancy/1",
        }
        assert validate_record(record) == []

    def test_missing_name_is_error(self) -> None:
        record = {
            "id": "v1",
            "name": "",
            "_phase0_source": "wayback-hhru",
            "_phase0_original_url": "https://hh.ru/vacancy/1",
        }
        errors = validate_record(record)
        assert any("name" in e.lower() for e in errors)

    def test_missing_name_key_is_error(self) -> None:
        record = {
            "id": "v1",
            "_phase0_source": "wayback-hhru",
            "_phase0_original_url": "https://hh.ru/vacancy/1",
        }
        errors = validate_record(record)
        assert any("name" in e.lower() for e in errors)

    def test_bad_source_is_error(self) -> None:
        record = {
            "id": "v1",
            "name": "Engineer",
            "_phase0_source": "unknown-source",
            "_phase0_original_url": "https://hh.ru/vacancy/1",
        }
        errors = validate_record(record)
        assert any("source" in e.lower() for e in errors)

    def test_each_valid_source_type_accepted(self) -> None:
        base = {
            "id": "v1",
            "name": "Engineer",
            "_phase0_original_url": "https://example.com",
        }
        for source in SOURCE_TYPES:
            record = {**base, "_phase0_source": source}
            assert validate_record(record) == [], f"Source {source!r} should be valid"

    def test_missing_original_url_is_error(self) -> None:
        record = {
            "id": "v1",
            "name": "Engineer",
            "_phase0_source": "wayback-hhru",
            "_phase0_original_url": "",
        }
        errors = validate_record(record)
        assert any("url" in e.lower() for e in errors)

    def test_missing_meta_fields_dont_crash_validation(self) -> None:
        """Validation is defensive: missing meta keys are treated as bad values."""
        record = {"id": "v1", "name": "Engineer"}
        errors = validate_record(record)
        # Should report source error, not crash
        assert len(errors) > 0


# ======================================================================
# Round-trip: raw → mapper → normalize → validate
# ======================================================================


class TestRoundTrip:
    """End-to-end: raw dict through mapper, normaliser, and validator."""

    @pytest.mark.parametrize(
        "mapper_fn,raw,kwargs",
        [
            (
                map_hhru_wayback,
                {
                    "id": "12345",
                    "name": "Python Developer",
                    "description": "Code",
                    "employer": {"name": "Acme"},
                    "area": {"name": "Moscow"},
                    "key_skills": ["Python"],
                },
                {"source_url": "https://hh.ru/vacancy/12345", "capture_ts": "ts"},
            ),
            (
                map_linkedin_wayback,
                {
                    "name": "Software Engineer",
                    "employer": {"name": "Google"},
                    "area": {"name": "CA"},
                    "description": "Build",
                    "key_skills": [{"name": "Go"}],
                },
                {
                    "source_url": "https://linkedin.com/jobs2/view/98765",
                    "capture_ts": "ts",
                },
            ),
            (
                map_trudvsem,
                {
                    "id": "tv1",
                    "vacancy_name": "Программист",
                    "vacancy_description": "Писать код",
                    "company": {"name": "Яндекс"},
                    "region": {"name": "Москва"},
                    "salary_min": 100_000,
                    "salary_max": 200_000,
                    "key_skills": [{"name": "Python"}],
                    "url": "https://trudvsem.ru/v/1",
                },
                {"capture_ts": "ts"},
            ),
            (
                map_rostud,
                {
                    "id": "r99",
                    "Должность": "Лаборант",
                    "Работодатель": "НИИ",
                    "Регион": "Казань",
                    "Описание": "Анализы",
                    "url": "https://rostrud.gov.ru/r99",
                },
                {"capture_ts": "ts"},
            ),
        ],
        ids=["hhru_wayback", "linkedin_wayback", "trudvsem", "rostud"],
    )
    def test_round_trip_mapper_normalize_validate(
        self, mapper_fn, raw, kwargs
    ) -> None:
        """raw → mapper → normalize_record → validate_record must all pass."""
        mapped = mapper_fn(raw, **kwargs)
        normalized = normalize_record(mapped)
        errors = validate_record(normalized)

        assert errors == [], f"Validation errors: {errors}"
        assert normalized["name"]
        assert normalized["id"]
        assert normalized["_phase0_source"] in SOURCE_TYPES
        assert normalized["_phase0_original_url"]
