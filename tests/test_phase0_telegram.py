"""Tests for Telegram channel ingestion (parser, schema mapper, collector helper)."""

from __future__ import annotations

from krm.phase0.collectors.telegram import _message_to_record
from krm.phase0.parsers.telegram import parse_telegram_message
from krm.phase0.schema import SOURCE_TYPES, map_telegram, normalize_record, validate_record


# ======================================================================
# parse_telegram_message
# ======================================================================


class TestParseTelegramMessage:
    def test_marker_based_russian(self) -> None:
        text = (
            "#вакансия\n"
            "Вакансия: Научный сотрудник\n"
            "Компания: НИИ Физики\n"
            "Город: Москва\n"
            "от 100 000 до 150 000 руб.\n"
            "Проведение экспериментов."
        )
        parsed = parse_telegram_message(
            text, channel="sciencejobs", message_id=42, date="2024-06-01"
        )

        assert parsed["name"] == "Научный сотрудник"
        assert parsed["employer"] == {"name": "НИИ Физики"}
        assert parsed["area"] == {"name": "Москва"}
        assert set(parsed["salary"]) == {"from", "to", "currency"}
        assert parsed["published_at"] == "2024-06-01"
        assert parsed["_source_url"] == "https://t.me/sciencejobs/42"
        assert parsed["key_skills"] == []
        assert parsed["experience"] == {"name": None}
        assert parsed["professional_roles"] == []
        assert "Проведение экспериментов" in parsed["description"]

    def test_freeform_english_first_line_as_name(self) -> None:
        text = (
            "Data Scientist\n"
            "Company: Google\n"
            "Location: Remote\n"
            "Build ML models."
        )
        parsed = parse_telegram_message(
            text, channel="jobs", message_id=7, date=None
        )

        assert parsed["name"] == "Data Scientist"
        assert parsed["employer"] == {"name": "Google"}
        assert parsed["area"] == {"name": "Remote"}
        assert parsed["published_at"] is None

    def test_no_marker_message(self) -> None:
        text = "Ищем лаборанта в химическую лабораторию."
        parsed = parse_telegram_message(
            text, channel="lab", message_id=1, date="2024-01-02"
        )

        assert parsed["name"] == "Ищем лаборанта в химическую лабораторию."
        assert parsed["employer"] == {"name": None}
        assert parsed["area"] == {"name": None}
        assert parsed["published_at"] == "2024-01-02"

    def test_never_raises_on_empty_text(self) -> None:
        parsed = parse_telegram_message("", channel="c", message_id=1, date=None)
        assert parsed["name"] == ""
        assert parsed["_source_url"] == "https://t.me/c/1"


# ======================================================================
# map_telegram
# ======================================================================


class TestMapTelegram:
    def test_source_and_id_prefix(self) -> None:
        raw = parse_telegram_message(
            "#вакансия\nДолжность: Химик",
            channel="lab",
            message_id=99,
            date="2024-06-01",
        )
        mapped = map_telegram(raw, "2024-06-02T00:00:00")

        assert mapped["_phase0_source"] == "telegram"
        assert mapped["id"] == "telegram-lab-99"
        assert mapped["_phase0_original_url"] == "https://t.me/lab/99"
        assert mapped["_phase0_capture_ts"] == "2024-06-02T00:00:00"

    def test_maps_fields_like_trudvsem(self) -> None:
        raw = {
            "name": "Химик",
            "description": "Синтез соединений",
            "employer": {"name": "НИИ"},
            "area": {"name": "Казань"},
            "salary": {"from": 100, "to": 200, "currency": "RUR"},
            "published_at": "2024-06-01",
            "key_skills": [{"name": "Химия"}],
            "_source_url": "https://t.me/lab/5",
        }
        mapped = map_telegram(raw, "ts")

        assert mapped["name"] == "Химик"
        assert mapped["description"] == "Синтез соединений"
        assert mapped["employer"] == {"name": "НИИ"}
        assert mapped["area"] == {"name": "Казань"}
        assert mapped["salary"] == {"from": 100, "to": 200, "currency": "RUR"}
        assert mapped["published_at"] == "2024-06-01"
        assert mapped["key_skills"] == [{"name": "Химия"}]
        assert mapped["id"] == "telegram-lab-5"

    def test_source_registered(self) -> None:
        assert "telegram" in SOURCE_TYPES

    def test_validates_after_normalize(self) -> None:
        raw = parse_telegram_message(
            "Должность: Инженер",
            channel="eng",
            message_id=3,
            date="2024-05-01",
        )
        mapped = map_telegram(raw, "2024-05-01T10:00:00")
        normalized = normalize_record(mapped)
        assert validate_record(normalized) == []


# ======================================================================
# _message_to_record (collector helper — no network)
# ======================================================================


class TestMessageToRecord:
    def test_full_pipeline_no_network(self) -> None:
        text = (
            "#вакансия\n"
            "Должность: Химик-аналитик\n"
            "Компания: НИИ\n"
            "Город: Москва\n"
        )
        record = _message_to_record(
            text, "lab", 123, "2024-06-01", "2024-06-01T10:00:00"
        )

        assert record["_phase0_source"] == "telegram"
        assert record["id"] == "telegram-lab-123"
        assert record["name"] == "Химик-аналитик"
        assert record["employer"] == {"name": "НИИ"}
        assert record["area"] == {"name": "Москва"}
        assert record["_phase0_original_url"] == "https://t.me/lab/123"
        assert record["_derived_year"] == 2024
        assert record["key_skills"] == []
