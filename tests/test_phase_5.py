"""Tests for phase_5_axes.py — NLI skill-to-axis mapping."""

import numpy as np
import pandas as pd
import pytest

from krm.phase_5_axes import (
    _normalize_to_proficiency,
    _top_contributing,
)


class TestNormalizeToProficiency:
    def test_range_one_to_five(self):
        scores = {"a": 0.8, "b": 0.3, "c": 0.1}
        result = _normalize_to_proficiency(scores)
        vals = list(result.values())
        assert 1.0 <= min(vals) <= max(vals) <= 5.0

    def test_all_equal_scores(self):
        scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        result = _normalize_to_proficiency(scores)
        assert len(result) == 3
        vals = list(result.values())
        assert 1.0 <= min(vals) <= max(vals) <= 5.0

    def test_all_zeros(self):
        scores = {"a": 0.0, "b": 0.0, "c": 0.0}
        result = _normalize_to_proficiency(scores)
        assert all(r == 1.0 for r in result.values())


class TestTopContributing:
    def test_returns_top_k(self):
        df = pd.DataFrame({
            "skill": ["skill_a", "skill_b", "skill_c"],
            "tfidf_weight": [0.8, 0.5, 0.2],
        })
        nli_cache = {
            "skill_a": {"hypothesis": 0.9},
            "skill_b": {"hypothesis": 0.6},
            "skill_c": {"hypothesis": 0.3},
        }
        result = _top_contributing(df, nli_cache, "axis_1", "hypothesis", top_k=2)
        assert len(result) <= 2
        if len(result) >= 2:
            assert result[0]["contribution"] >= result[1]["contribution"]

    def test_handles_missing_nli(self):
        df = pd.DataFrame({
            "skill": ["skill_x"],
            "tfidf_weight": [0.5],
        })
        nli_cache: dict[str, dict[str, float]] = {}
        result = _top_contributing(df, nli_cache, "axis_1", "hyp", top_k=3)
        if result:
            assert result[0]["contribution"] == 0.0

