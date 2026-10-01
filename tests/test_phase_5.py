"""Tests for phase_5_axes.py — taxonomy-driven skill-to-characteristic mapping."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from krm import phase_5_axes
from krm.config import Config
from krm.lib.llm import LLMClient

_CONTENT_AXES = [
    "domain_knowledge",
    "experimental",
    "data_analysis",
    "computational",
    "professional_texts",
    "t_profile",
    "management",
]
_MATRIX_COLUMNS = ["skill_canonical_name", "characteristic_id", "nli_score"]

_TOKEN_ORDER = ["domain", "experiment", "data", "compute", "texts", "manage", "outlook"]

_GLOSSARIES = {
    "domain_knowledge": ["domain"],
    "experimental": ["experiment"],
    "data_analysis": ["data"],
    "computational": ["compute"],
    "professional_texts": ["texts"],
    "management": ["manage"],
    "t_profile": ["outlook"],
}


def _unit_vector(i: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


class _LookupEmbedder:
    """One-hot embedder: a text embedding is a unit basis vector when it names an axis."""

    def __init__(self, model_name: str, cache_dir=None, device=None) -> None:
        self._vocab = {
            token: _unit_vector(i, len(_TOKEN_ORDER))
            for i, token in enumerate(_TOKEN_ORDER)
        }

    def encode(self, texts, batch_size: int = 32, **kwargs) -> np.ndarray:
        dim = len(_TOKEN_ORDER)
        return np.array(
            [self._vocab.get(t, np.zeros(dim, dtype=np.float32)) for t in texts],
            dtype=np.float32,
        )


@pytest.fixture
def fake_config(tmp_path, monkeypatch) -> Config:
    """Config writing to a temp dir with a deterministic embedder and glossaries."""
    config = Config()
    monkeypatch.setattr(Config, "output_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(
        Config, "characteristic_seed_glossaries", property(lambda self: _GLOSSARIES)
    )
    monkeypatch.setattr(phase_5_axes, "Embedder", _LookupEmbedder)
    return config


def _skills_df() -> pd.DataFrame:
    """Role 0 → experimental specialist; role 1 → balanced generalist with Кругозор."""
    return pd.DataFrame({
        "role_id": [0, 1, 1, 1, 1, 1, 1],
        "skill_canonical_name": [
            "experiment", "domain", "data", "compute", "texts", "manage", "outlook",
        ],
        "tfidf_weight": [0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    })


class TestSkillTagMatrix:
    def test_tag_exact_match_case_insensitive(self):
        tags = phase_5_axes._skill_tag_matrix(["EXPERIMENT", "unknown"], _CONTENT_AXES, _GLOSSARIES)
        assert tags[0, _CONTENT_AXES.index("experimental")] == 1.0
        assert tags[0, _CONTENT_AXES.index("t_profile")] == 0.0
        assert tags[1].sum() == 0.0

    def test_multi_axis_tag_normalizes(self):
        glossaries = dict(_GLOSSARIES)
        glossaries["experimental"] = ["experiment", "shared"]
        glossaries["data_analysis"] = ["data", "shared"]
        tags = phase_5_axes._skill_tag_matrix(["shared"], _CONTENT_AXES, glossaries)
        assert tags[0, _CONTENT_AXES.index("experimental")] == pytest.approx(0.5)
        assert tags[0, _CONTENT_AXES.index("data_analysis")] == pytest.approx(0.5)
        assert tags[0].sum() == pytest.approx(1.0)


class TestMapToCharacteristics:
    def test_t_profile_is_scored_content_axis(self, fake_config):
        _skills_df().to_parquet(fake_config.output_dir / "skills_per_role.parquet", index=False)

        phase_5_axes.map_to_characteristics(fake_config)

        scores = pd.read_parquet(fake_config.characteristic_scores_path)
        assert set(scores["characteristic_id"]) == set(fake_config.characteristic_ids)
        assert len(scores) == 2 * 7  # 7 axes × 2 roles

        t_profile = scores[scores["characteristic_id"] == "t_profile"].set_index("role_id")
        t_profile = t_profile["proficiency"]
        # Both roles land in [1, 5] and are differentiated: role 0 has no
        # outlook skill (floor 1.0), role 1 has a tagged outlook skill (ceiling 5.0).
        assert t_profile.between(1.0, 5.0).all()
        assert t_profile[0] == pytest.approx(1.0, abs=1e-3)
        assert t_profile[1] == pytest.approx(5.0, abs=1e-3)
        assert t_profile[0] != pytest.approx(t_profile[1])

    def test_t_profile_has_top_contributing_skills(self, fake_config):
        _skills_df().to_parquet(fake_config.output_dir / "skills_per_role.parquet", index=False)

        phase_5_axes.map_to_characteristics(fake_config)

        scores = pd.read_parquet(fake_config.characteristic_scores_path)
        t_prof = scores[scores["characteristic_id"] == "t_profile"].set_index("role_id")
        top = json.loads(t_prof.loc[1, "top_contributing_skills"])
        assert top[0]["skill"] == "outlook"
        assert top[0]["contribution"] == pytest.approx(1.0, abs=1e-4)

    def test_taxonomy_tag_dominates_cosine(self, fake_config):
        _skills_df().to_parquet(fake_config.output_dir / "skills_per_role.parquet", index=False)

        phase_5_axes.map_to_characteristics(fake_config)

        matrix = pd.read_parquet(fake_config.skill_characteristic_scores_path)
        row = matrix[matrix["skill_canonical_name"] == "outlook"].set_index("characteristic_id")
        # tag = 1.0, cosine = 1.0 (one-hot) → 0.7 * 1 + 0.3 * 1
        assert row.loc["t_profile", "nli_score"] == pytest.approx(0.7 + 0.3, abs=1e-6)
        # untagged axis → 0.7 * 0 + 0.3 * 0 (orthogonal centroid)
        assert row.loc["experimental", "nli_score"] == pytest.approx(0.0, abs=1e-6)

    def test_all_axes_globally_normalized(self, fake_config):
        _skills_df().to_parquet(fake_config.output_dir / "skills_per_role.parquet", index=False)

        phase_5_axes.map_to_characteristics(fake_config)

        scores = pd.read_parquet(fake_config.characteristic_scores_path)
        content = scores[scores["characteristic_id"].isin(_CONTENT_AXES)]
        assert content["proficiency"].between(1.0, 5.0).all()
        # Global normalization must spread each axis across roles (no within-role pinning).
        for axis in _CONTENT_AXES:
            values = content[content["characteristic_id"] == axis]["proficiency"]
            assert values.min() == pytest.approx(1.0, abs=1e-3)
            assert values.max() == pytest.approx(5.0, abs=1e-3)

    def test_top_contributing_skills_present(self, fake_config):
        _skills_df().to_parquet(fake_config.output_dir / "skills_per_role.parquet", index=False)

        phase_5_axes.map_to_characteristics(fake_config)

        scores = pd.read_parquet(fake_config.characteristic_scores_path)
        exp = scores[scores["characteristic_id"] == "experimental"].set_index("role_id")
        top = json.loads(exp.loc[0, "top_contributing_skills"])
        assert [t["skill"] for t in top] == ["experiment"]

    def test_persists_skill_characteristic_scores_matrix(self, fake_config):
        _skills_df().to_parquet(fake_config.output_dir / "skills_per_role.parquet", index=False)

        phase_5_axes.map_to_characteristics(fake_config)

        matrix = pd.read_parquet(fake_config.skill_characteristic_scores_path)
        assert list(matrix.columns) == _MATRIX_COLUMNS
        assert set(matrix["characteristic_id"]) == set(_CONTENT_AXES)

    def test_empty_skills_writes_empty_outputs(self, fake_config):
        empty = pd.DataFrame({
            "role_id": pd.Series([], dtype="int64"),
            "skill_canonical_name": pd.Series([], dtype="object"),
            "tfidf_weight": pd.Series([], dtype="float64"),
        })
        empty.to_parquet(fake_config.output_dir / "skills_per_role.parquet", index=False)

        phase_5_axes.map_to_characteristics(fake_config)

        scores = pd.read_parquet(fake_config.characteristic_scores_path)
        assert scores.empty
        matrix = pd.read_parquet(fake_config.skill_characteristic_scores_path)
        assert matrix.empty
        assert list(matrix.columns) == _MATRIX_COLUMNS


def _skills_with_untagged() -> pd.DataFrame:
    """Two glossary skills + two out-of-glossary skills for LLM backfill."""
    return pd.DataFrame({
        "role_id": [0, 0, 0, 1],
        "skill_canonical_name": ["experiment", "llm_tagged", "llm_missing", "data"],
        "tfidf_weight": [0.9, 0.8, 0.7, 0.6],
    })


class TestLLMPhase5:
    """LLM backfill of glossary-untagged skills (no network, run_many stubbed)."""

    @pytest.fixture
    def llm_config(self, fake_config, tmp_path, monkeypatch) -> Config:
        monkeypatch.setattr(Config, "use_llm_phase_5", property(lambda self: True))
        monkeypatch.setattr(
            Config, "llm_cache_dir", property(lambda self: str(tmp_path / "llm_cache"))
        )
        return fake_config

    def test_llm_tags_only_out_of_glossary_skills(self, llm_config, monkeypatch):
        sent: list[str] = []

        async def fake_run_many(self, prompts, schema):
            assert schema is phase_5_axes.AxisTagResponse
            sent.extend(p.user for p in prompts)
            return [
                phase_5_axes.AxisTagResponse(axis="data_analysis")
                if "llm_tagged" in p.user
                else None
                for p in prompts
            ]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)
        _skills_with_untagged().to_parquet(
            llm_config.output_dir / "skills_per_role.parquet", index=False
        )

        phase_5_axes.map_to_characteristics(llm_config)

        # (a) only out-of-glossary skills are sent to the LLM.
        sent_skills = [u.removeprefix("Навык: ") for u in sent]
        assert set(sent_skills) == {"llm_tagged", "llm_missing"}

        # (b) LLM tags merge into the matrix (tag=1, cosine=0 → 0.7).
        matrix = pd.read_parquet(llm_config.skill_characteristic_scores_path)
        tagged = matrix[
            matrix["skill_canonical_name"] == "llm_tagged"
        ].set_index("characteristic_id")
        assert tagged.loc["data_analysis", "nli_score"] == pytest.approx(0.7, abs=1e-6)

        # (c) None → left untagged (deterministic all-zero score).
        missing = matrix[
            matrix["skill_canonical_name"] == "llm_missing"
        ].set_index("characteristic_id")
        assert (missing["nli_score"] == 0.0).all()

    def test_llm_does_not_retag_glossary_skills(self, llm_config, monkeypatch):
        sent: list[str] = []

        async def fake_run_many(self, prompts, schema):
            sent.extend(p.user for p in prompts)
            return [phase_5_axes.AxisTagResponse(axis="management") for _ in prompts]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)
        _skills_with_untagged().to_parquet(
            llm_config.output_dir / "skills_per_role.parquet", index=False
        )

        phase_5_axes.map_to_characteristics(llm_config)

        # Glossary skills are never sent; their deterministic tag is preserved.
        assert "experiment" not in sent and "data" not in sent
        matrix = pd.read_parquet(llm_config.skill_characteristic_scores_path)
        exp = matrix[
            matrix["skill_canonical_name"] == "experiment"
        ].set_index("characteristic_id")
        # experiment → tag 1.0 on experimental, cosine 1.0 → 0.7 + 0.3.
        assert exp.loc["experimental", "nli_score"] == pytest.approx(1.0, abs=1e-6)

    def test_llm_disabled_is_deterministic(self, fake_config):
        """Default config (use_llm_phase_5 False) never touches the LLM."""
        _skills_with_untagged().to_parquet(
            fake_config.output_dir / "skills_per_role.parquet", index=False
        )

        phase_5_axes.map_to_characteristics(fake_config)

        matrix = pd.read_parquet(fake_config.skill_characteristic_scores_path)
        tagged = matrix[
            matrix["skill_canonical_name"] == "llm_tagged"
        ].set_index("characteristic_id")
        assert (tagged["nli_score"] == 0.0).all()
