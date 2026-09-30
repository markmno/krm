"""Tests for phase_4_skills.py — data-driven role-level skill aggregation.

Tests the pure aggregation logic (no torch, sentence-transformers, hdbscan,
or network required). ESCO was deprecated; the matching helpers these tests
used to cover are gone.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from krm.phase_4_skills import _aggregate_skills, extract_skills

_OUTPUT_COLUMNS = ["role_id", "skill_canonical_name", "frequency", "tfidf_weight"]


def _extracted_df() -> pd.DataFrame:
    """A small Phase 2b-style vacancy-level skills frame."""
    return pd.DataFrame(
        {
            "vacancy_id": ["v1", "v1", "v2", "v3", "v4"],
            "skill_phrase": ["python", "ml", "python", "ml", "python"],
            "frequency": [1, 1, 1, 1, 1],
            "tfidf_weight": [0.5, 0.3, 0.4, 0.2, 0.9],
        }
    )


class TestAggregateSkills:
    def test_sums_frequency_and_tfidf_per_role_skill(self):
        """Frequency and TF-IDF are summed across vacancies in the same role."""
        df = _extracted_df()
        role_map = pd.Series({"v1": 0, "v2": 0, "v3": 1, "v4": 1})
        out = _aggregate_skills(df, role_map)

        row = out[(out["role_id"] == 0) & (out["skill_canonical_name"] == "python")]
        assert len(row) == 1
        assert row.iloc[0]["frequency"] == 2
        assert row.iloc[0]["tfidf_weight"] == pytest.approx(0.9)  # 0.5 + 0.4

    def test_drops_noise_vacancies(self):
        """Vacancies assigned to the noise role (-1) are excluded."""
        df = _extracted_df()
        role_map = pd.Series({"v1": 0, "v2": -1, "v3": 0, "v4": -1})
        out = _aggregate_skills(df, role_map)

        assert -1 not in set(out["role_id"])
        # v2 (python) and v4 (python) were noise → only v1 remains for python.
        row = out[(out["role_id"] == 0) & (out["skill_canonical_name"] == "python")]
        assert row.iloc[0]["frequency"] == 1

    def test_drops_unmapped_vacancies(self):
        """Vacancies with no role mapping (NaN) are excluded."""
        df = _extracted_df()
        role_map = pd.Series({"v1": 0, "v3": 0})  # v2, v4 unmapped
        out = _aggregate_skills(df, role_map)
        assert set(out["skill_canonical_name"]) == {"python", "ml"}

    def test_output_columns(self):
        out = _aggregate_skills(_extracted_df(), pd.Series({"v1": 0}))
        assert list(out.columns) == _OUTPUT_COLUMNS

    def test_role_id_is_integer(self):
        out = _aggregate_skills(_extracted_df(), pd.Series({"v1": 0}))
        assert pd.api.types.is_integer_dtype(out["role_id"])

    def test_all_noise_yields_empty_with_columns(self):
        df = _extracted_df()
        role_map = pd.Series({"v1": -1, "v2": -1, "v3": -1, "v4": -1})
        out = _aggregate_skills(df, role_map)
        assert out.empty
        assert list(out.columns) == _OUTPUT_COLUMNS

    def test_multiple_skills_same_role(self):
        """A role with several distinct skills produces one row per skill."""
        df = _extracted_df()
        role_map = pd.Series({"v1": 0, "v2": 0, "v3": 0, "v4": 0})
        out = _aggregate_skills(df, role_map)
        assert set(out["skill_canonical_name"]) == {"python", "ml"}
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Tests: extract_skills persists vacancy_roles.parquet (T2)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic embedder — avoids sentence-transformers/torch in tests."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, sum(ord(c) for c in text) % self._dim] = 1.0
        return out


def _make_config(tmp_path: Path) -> MagicMock:
    config = MagicMock()
    type(config).output_dir = PropertyMock(return_value=tmp_path)
    type(config).embedding_model = PropertyMock(return_value="dummy-model")
    type(config).vacancy_roles_path = PropertyMock(
        return_value=tmp_path / "vacancy_roles.parquet"
    )
    return config


def _write_roles_parquet(tmp_path: Path) -> None:
    centroids = {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0, 0.0]}
    rows = [
        {
            "role_id": rid,
            "centroid_embedding": pickle.dumps(np.array(vec, dtype=np.float32)),
        }
        for rid, vec in centroids.items()
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "roles.parquet", index=False)


def _write_classified_parquet(tmp_path: Path, *, stem: bool) -> None:
    rows = [
        {
            "vacancy_id": f"v{i}",
            "title": f"title-{i}",
            "description": f"desc-{i}",
            "stem_category": "STEM_RESEARCH" if stem else "NON_STEM",
        }
        for i in range(4)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "classified.parquet", index=False)


def _write_extracted_parquet(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "vacancy_id": ["v0", "v0", "v1", "v2", "v3"],
            "skill_phrase": ["python", "ml", "python", "ml", "stats"],
            "frequency": [1, 1, 1, 1, 1],
            "tfidf_weight": [0.5, 0.3, 0.4, 0.2, 0.9],
        }
    ).to_parquet(tmp_path / "extracted_skills.parquet", index=False)


class TestPersistVacancyRoles:
    """extract_skills must persist the vacancy→role map as vacancy_roles.parquet."""

    def test_writes_vacancy_roles_parquet(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_roles_parquet(tmp_path)
        _write_classified_parquet(tmp_path, stem=True)
        _write_extracted_parquet(tmp_path)

        with patch("krm.phase_4_skills.Embedder", return_value=_FakeEmbedder()):
            extract_skills(config)

        out = pd.read_parquet(tmp_path / "vacancy_roles.parquet")
        assert list(out.columns) == ["vacancy_id", "role_id"]
        assert (out["role_id"] != -1).all()
        assert out["vacancy_id"].is_unique
        assert pd.api.types.is_string_dtype(out["vacancy_id"])
        assert pd.api.types.is_integer_dtype(out["role_id"])

    def test_empty_stem_writes_empty_vacancy_roles(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_roles_parquet(tmp_path)
        _write_classified_parquet(tmp_path, stem=False)

        extract_skills(config)  # no embedder needed — _empty short-circuits

        out = pd.read_parquet(tmp_path / "vacancy_roles.parquet")
        assert out.empty
        assert list(out.columns) == ["vacancy_id", "role_id"]
