"""Tests for phase_5b_soft.py — archetype classification + rubric-derived soft scoring."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from krm import phase_5b_soft
from krm.config import Config
from krm.lib.llm import LLMClient
from krm.phase_5b_soft import SoftScore, SoftScoresResponse

_SOFT_IDS = ["thinking", "teamwork", "leadership", "professional_culture"]
_CONTENT_AXES = [
    "domain_knowledge",
    "experimental",
    "data_analysis",
    "computational",
    "professional_texts",
    "management",
]


@pytest.fixture
def fake_config(tmp_path, monkeypatch) -> Config:
    """Config writing to a temp dir."""
    config = Config()
    monkeypatch.setattr(Config, "output_dir", property(lambda self: tmp_path))
    return config


@pytest.fixture
def llm_config(fake_config, monkeypatch) -> Config:
    """Config with the LLM phase flag on and the on-disk cache disabled."""
    monkeypatch.setattr(Config, "use_llm_phase_5b", property(lambda self: True))
    monkeypatch.setattr(Config, "llm_cache_dir", property(lambda self: None))
    return fake_config


def _skills_df() -> pd.DataFrame:
    """Role 0 → experimental (Техник); role 1 → computational (Инженер)."""
    return pd.DataFrame({
        "role_id": [0, 0, 0, 0, 1, 1, 1],
        "skill_canonical_name": [
            "лабораторный", "эксперимент", "спектрометр", "хроматограф",
            "python", "программирование", "алгоритм",
        ],
        "tfidf_weight": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7],
    })


_HARD_SCORES = {
    0: {
        "domain_knowledge": 1.0,
        "experimental": 2.0,
        "data_analysis": 3.0,
        "computational": 4.0,
        "professional_texts": 5.0,
        "management": 2.0,
        "t_profile": 3.0,
    },
    1: {
        "domain_knowledge": 5.0,
        "experimental": 1.0,
        "data_analysis": 2.0,
        "computational": 2.0,
        "professional_texts": 3.0,
        "management": 4.0,
        "t_profile": 4.0,
    },
}


def _write_characteristic_scores(config: Config, hard_scores: dict[int, dict[str, float]]) -> None:
    rows = []
    for role_id, axes in hard_scores.items():
        for characteristic_id, proficiency in axes.items():
            rows.append({
                "role_id": role_id,
                "characteristic_id": characteristic_id,
                "proficiency": proficiency,
                "top_contributing_skills": json.dumps([], ensure_ascii=False),
            })
    pd.DataFrame(rows).to_parquet(config.characteristic_scores_path, index=False)


class TestClassifyRoleArchetypes:
    def test_writes_archetypes_from_top_skills(self, fake_config):
        _skills_df().to_parquet(fake_config.skills_per_role_path, index=False)

        out = phase_5b_soft.classify_role_archetypes(fake_config)

        assert list(out.columns) == ["role_id", "archetype", "super_fractions"]

        stored = pd.read_parquet(fake_config.role_archetypes_path)
        assert set(stored["archetype"]) <= set(fake_config.archetype_names)
        assert stored["archetype"].astype(str).str.len().gt(0).all()

        mapping = dict(zip(stored["role_id"], stored["archetype"], strict=True))
        assert mapping[0] == "Техник"
        assert mapping[1] == "Инженер"

    def test_empty_skills_writes_empty_parquet(self, fake_config):
        empty = pd.DataFrame({
            "role_id": pd.Series([], dtype="int64"),
            "skill_canonical_name": pd.Series([], dtype="object"),
            "tfidf_weight": pd.Series([], dtype="float64"),
        })
        empty.to_parquet(fake_config.skills_per_role_path, index=False)

        out = phase_5b_soft.classify_role_archetypes(fake_config)

        assert out.empty
        assert list(out.columns) == ["role_id", "archetype", "super_fractions"]


class TestScoreSoftCompetences:
    def test_soft_equals_rubric_applied_to_hard_scores(self, fake_config):
        _write_characteristic_scores(fake_config, _HARD_SCORES)

        out = phase_5b_soft.score_soft_competences(fake_config)

        assert list(out.columns) == ["role_id", "soft_id", "proficiency"]
        stored = pd.read_parquet(fake_config.soft_scores_path)
        assert set(stored["soft_id"]) == set(_SOFT_IDS)

        rubric = fake_config.soft_rubric
        for role_id, hard in _HARD_SCORES.items():
            for soft_id in _SOFT_IDS:
                expected = sum(
                    hard[axis] * weight
                    for axis, weight in rubric[soft_id].items()
                )
                actual = stored[
                    (stored["role_id"] == role_id) & (stored["soft_id"] == soft_id)
                ]["proficiency"].iloc[0]
                assert actual == pytest.approx(expected, abs=1e-4)

        assert stored["proficiency"].between(1.0, 5.0).all()

    def test_rubric_rows_sum_to_one(self, fake_config):
        rubric = fake_config.soft_rubric
        assert set(rubric) == set(_SOFT_IDS)
        for weights in rubric.values():
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_empty_characteristic_scores_writes_empty(self, fake_config):
        empty = pd.DataFrame({
            "role_id": pd.Series([], dtype="int64"),
            "characteristic_id": pd.Series([], dtype="object"),
            "proficiency": pd.Series([], dtype="float64"),
            "top_contributing_skills": pd.Series([], dtype="object"),
        })
        empty.to_parquet(fake_config.characteristic_scores_path, index=False)

        out = phase_5b_soft.score_soft_competences(fake_config)

        assert out.empty
        assert list(out.columns) == ["role_id", "soft_id", "proficiency"]


class TestSoftScoreModel:
    def test_proficiency_is_clamped_to_1_5(self) -> None:
        assert SoftScore(soft_id="x", proficiency=9.0).proficiency == 5.0
        assert SoftScore(soft_id="x", proficiency=0.0).proficiency == 1.0
        assert SoftScore(soft_id="x", proficiency=3.5).proficiency == 3.5


class TestScoreSoftCompetencesLlm:
    def test_llm_scores_roles_and_clamps(self, llm_config, monkeypatch) -> None:
        _write_characteristic_scores(llm_config, _HARD_SCORES)
        _skills_df().to_parquet(llm_config.skills_per_role_path, index=False)

        captured: dict[str, object] = {}

        async def fake_run_many(self, prompts, schema):
            captured["schema"] = schema
            captured["prompts"] = prompts
            return [
                SoftScoresResponse(scores=[
                    SoftScore(soft_id="thinking", proficiency=4.5),
                    SoftScore(soft_id="teamwork", proficiency=2.0),
                    SoftScore(soft_id="leadership", proficiency=9.0),  # clamped to 5.0
                    SoftScore(soft_id="professional_culture", proficiency=3.25),
                    SoftScore(soft_id="bogus_axis", proficiency=3.0),  # unknown → dropped
                ]),
                SoftScoresResponse(scores=[
                    SoftScore(soft_id="thinking", proficiency=1.0),
                    SoftScore(soft_id="teamwork", proficiency=1.5),
                    SoftScore(soft_id="leadership", proficiency=2.0),
                    SoftScore(soft_id="professional_culture", proficiency=2.5),
                ]),
            ]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)

        out = phase_5b_soft.score_soft_competences(llm_config)

        assert captured["schema"] is SoftScoresResponse
        assert len(captured["prompts"]) == 2
        assert all(soft_id in captured["prompts"][0].user for soft_id in _SOFT_IDS)
        assert "лабораторный" in captured["prompts"][0].user

        assert list(out.columns) == ["role_id", "soft_id", "proficiency"]
        stored = pd.read_parquet(llm_config.soft_scores_path)
        got = {
            (int(row["role_id"]), row["soft_id"]): float(row["proficiency"])
            for _, row in stored.iterrows()
        }
        expected = {
            (0, "thinking"): 4.5,
            (0, "teamwork"): 2.0,
            (0, "leadership"): 5.0,
            (0, "professional_culture"): 3.25,
            (1, "thinking"): 1.0,
            (1, "teamwork"): 1.5,
            (1, "leadership"): 2.0,
            (1, "professional_culture"): 2.5,
        }
        assert got == expected
        assert stored["proficiency"].between(1.0, 5.0).all()

    def test_none_falls_back_to_rubric(self, llm_config, monkeypatch) -> None:
        _write_characteristic_scores(llm_config, _HARD_SCORES)
        _skills_df().to_parquet(llm_config.skills_per_role_path, index=False)

        async def fake_run_many(self, prompts, schema):
            return [
                SoftScoresResponse(scores=[
                    SoftScore(soft_id="thinking", proficiency=4.0),
                    SoftScore(soft_id="teamwork", proficiency=3.0),
                    SoftScore(soft_id="leadership", proficiency=2.0),
                    SoftScore(soft_id="professional_culture", proficiency=1.5),
                ]),
                None,  # role 1 → rubric fallback
            ]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)

        phase_5b_soft.score_soft_competences(llm_config)
        stored = pd.read_parquet(llm_config.soft_scores_path)
        got = {
            (int(row["role_id"]), row["soft_id"]): float(row["proficiency"])
            for _, row in stored.iterrows()
        }

        assert got[(0, "thinking")] == pytest.approx(4.0)

        rubric = llm_config.soft_rubric
        for soft_id in _SOFT_IDS:
            expected = sum(
                _HARD_SCORES[1][axis] * weight
                for axis, weight in rubric[soft_id].items()
            )
            assert got[(1, soft_id)] == pytest.approx(expected, abs=1e-4)

    def test_use_llm_false_never_calls_llm(self, fake_config, monkeypatch) -> None:
        _write_characteristic_scores(fake_config, _HARD_SCORES)
        _skills_df().to_parquet(fake_config.skills_per_role_path, index=False)

        async def explode(self, prompts, schema):
            raise AssertionError("LLM path must not be used when use_llm_phase_5b is false")

        monkeypatch.setattr(LLMClient, "run_many", explode)

        out = phase_5b_soft.score_soft_competences(fake_config)

        assert list(out.columns) == ["role_id", "soft_id", "proficiency"]
        rubric = fake_config.soft_rubric
        stored = pd.read_parquet(fake_config.soft_scores_path)
        for role_id, hard in _HARD_SCORES.items():
            for soft_id in _SOFT_IDS:
                expected = sum(
                    hard[axis] * weight
                    for axis, weight in rubric[soft_id].items()
                )
                actual = stored[
                    (stored["role_id"] == role_id) & (stored["soft_id"] == soft_id)
                ]["proficiency"].iloc[0]
                assert actual == pytest.approx(expected, abs=1e-4)
