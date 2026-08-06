"""Tests for phase_2_classify.py — STEM/IT/NON_STEM classifier."""

import pytest

from krm.phase_2_classify import (
    STEM_RESEARCH,
    PURE_IT,
    NON_STEM,
    INTERDISCIPLINARY,
    _tier1_is_pure_it,
    _tier2_is_pure_it,
    _build_nli_text,
)


class TestConstants:
    def test_category_constants_are_distinct(self):
        categories = {STEM_RESEARCH, PURE_IT, NON_STEM, INTERDISCIPLINARY}
        assert len(categories) == 4

    def test_valid_categories(self):
        assert STEM_RESEARCH == "STEM_RESEARCH"
        assert PURE_IT == "PURE_IT"


class TestTier1:
    def test_it_role_is_pure_it(self):
        assert _tier1_is_pure_it([96]) is True
        assert _tier1_is_pure_it([123]) is True
        assert _tier1_is_pure_it([125]) is True

    def test_rd_role_is_not_it(self):
        assert _tier1_is_pure_it([79]) is False
        assert _tier1_is_pure_it([]) is False


class TestTier2:
    def test_developer_without_stem_is_pure_it(self):
        assert _tier2_is_pure_it("Программист Python", "разработка веб-приложений") is True

    def test_developer_with_stem_is_not_pure_it(self):
        assert _tier2_is_pure_it(
            "Разработчик",
            "разработка лабораторного оборудования для научных исследований",
        ) is False

    def test_scientist_not_caught(self):
        assert _tier2_is_pure_it("Научный сотрудник", "экспериментальная физика") is False


class TestNliText:
    def test_build_nli_text_with_both(self):
        text = _build_nli_text("Физик", "экспериментальная физика")
        assert "Физик" in text
        assert "экспериментальная физика" in text

    def test_build_nli_text_title_only(self):
        text = _build_nli_text("Химик", None)
        assert "Химик" in text
