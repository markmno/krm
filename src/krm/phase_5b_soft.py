"""Phase 5b: Role Archetype Classification and Soft-Competence Scoring.

Two-stage pipeline run after Phase 4 (skills) and Phase 5 (hard axes):

1. :func:`classify_role_archetypes` — classifies each discovered role into one
   of six ISCB-inspired archetypes (Техник, Исследователь, Методолог, Инженер,
   Ведущий специалист, Гибрид) from its top-20 defining skills.
2. :func:`score_soft_competences` — derives each role's four soft competences
   from its six hard-axis scores via a fixed linear rubric (default) or the
   shared LLM client when ``config.use_llm_phase_5b`` is true.

Usage::

    from krm.config import Config
    from krm.phase_5b_soft import classify_role_archetypes, score_soft_competences

    config = Config()
    classify_role_archetypes(config)
    soft_scores = score_soft_competences(config)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, field_validator

from krm.analysis.archetypes import classify_archetypes
from krm.config import Config
from krm.lib.io import read_parquet, write_parquet
from krm.lib.llm import LLMPrompt, build_llm

_TOP_DEFINING_SKILLS = 20
_PROF_MIN = 1.0
_PROF_MAX = 5.0


@dataclass(frozen=True)
class _RoleAdapter:
    """Structural adapter satisfying ``classify_archetypes``' role protocol."""

    role_name: str
    defining_skills: list[tuple[str, float]]


def classify_role_archetypes(config: Config) -> pd.DataFrame:
    """Classify each role into an archetype from its top-20 defining skills.

    Reads ``skills_per_role.parquet``, keeps the top-20 skills per role by
    TF-IDF weight, adapts them into the ``_HasDefiningSkills`` protocol, and
    writes ``role_archetypes.parquet``.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame with columns ``role_id`` (int), ``archetype`` (str), and
        ``super_fractions`` (JSON str). Empty if no skills exist.
    """
    skills_df = read_parquet(config.skills_per_role_path)

    empty_columns = {
        "role_id": pd.Series([], dtype="int64"),
        "archetype": pd.Series([], dtype="object"),
        "super_fractions": pd.Series([], dtype="object"),
    }
    if skills_df.empty:
        empty = pd.DataFrame(empty_columns)
        write_parquet(empty, config.role_archetypes_path)
        return empty

    roles: list[_RoleAdapter] = []
    role_ids: list[int] = []
    for role_id, grp in skills_df.groupby("role_id", sort=True):
        top = grp.sort_values("tfidf_weight", ascending=False).head(_TOP_DEFINING_SKILLS)
        defining_skills = [
            (str(row["skill_canonical_name"]), float(row["tfidf_weight"]))
            for _, row in top.iterrows()
        ]
        roles.append(_RoleAdapter(role_name=str(int(role_id)), defining_skills=defining_skills))
        role_ids.append(int(role_id))

    classifications = classify_archetypes(roles, discovered_axes=[])

    archetype_col: list[str] = []
    super_col: list[str] = []
    for classification in classifications:
        archetype_col.append(classification.archetype.name)
        super_col.append(json.dumps(classification.super_fractions, ensure_ascii=False))

    out = pd.DataFrame({
        "role_id": pd.Series(role_ids, dtype="int64"),
        "archetype": archetype_col,
        "super_fractions": super_col,
    })
    write_parquet(out, config.role_archetypes_path)
    print(f"[Phase 5b] Wrote {len(out)} role archetypes to {config.role_archetypes_path}")
    return out


def _apply_rubric(
    hard_scores: dict[str, float],
    rubric: dict[str, dict[str, float]],
    soft_ids: list[str],
) -> dict[str, float]:
    """Compute raw soft scores as the rubric linear combination of hard scores."""
    result: dict[str, float] = {}
    for soft_id in soft_ids:
        weights = rubric.get(soft_id, {})
        result[soft_id] = sum(
            hard_scores.get(hard, 0.0) * weight for hard, weight in weights.items()
        )
    return result


_LLM_SOFT_SYSTEM_PROMPT = (
    "Ты — эксперт по анализу профессиональных профилей научных и инженерных "
    "ролей. На основе списка ключевых навыков роли оцени уровень каждой из "
    "перечисленных ниже мягких компетенций. Верни строго JSON со списком "
    "оценок: для каждой компетенции укажи её идентификатор (soft_id) и уровень "
    "владения (proficiency) в диапазоне от 1.0 до 5.0, где 1 — минимальный "
    "уровень, а 5 — максимальный. Используй только идентификаторы из списка."
)


class SoftScore(BaseModel):
    """A single soft-competence proficiency score produced by the LLM."""

    soft_id: str
    proficiency: float

    @field_validator("proficiency")
    @classmethod
    def _clamp_proficiency(cls, value: float) -> float:
        """Clamp the self-assessed proficiency into [1, 5]."""
        return min(max(value, _PROF_MIN), _PROF_MAX)


class SoftScoresResponse(BaseModel):
    """LLM structured output: proficiency per soft axis for one role.

    Entries whose ``soft_id`` is not a configured soft axis are ignored by the
    caller — the allow-list lives in config, not in the schema, so it is
    enforced at the dispatch boundary in :func:`_score_soft_llm`.
    """

    scores: list[SoftScore]


def _score_soft_llm(
    config: Config,
    role_hard: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Score each role's soft competences via the shared LLM client.

    Reads ``skills_per_role.parquet``, keeps the top-20 skills per role by
    TF-IDF weight (the same grouping as :func:`classify_role_archetypes`),
    builds one prompt per role listing those skills plus the four soft-axis
    definitions, and runs them in one batch. A role whose LLM call fails
    (``None``) falls back to the rubric result for that role only.

    Args:
        config: Pipeline configuration.
        role_hard: Mapping ``role_id → {characteristic_id: proficiency}`` from
            ``characteristic_scores.parquet``, used for the rubric fallback.

    Returns:
        Mapping ``role_id → {soft_id: proficiency}`` covering all four soft
        axes per role.
    """
    soft_ids = config.soft_competence_ids
    rubric = config.soft_rubric
    labels_ru = config.soft_competence_labels_ru
    hypotheses = config.soft_competence_hypotheses
    definitions = "\n".join(
        f"- {soft_id} ({labels_ru[soft_id]}): {hypotheses[soft_id]}"
        for soft_id in soft_ids
    )

    skills_df = read_parquet(config.skills_per_role_path)

    roles: list[int] = []
    prompts: list[LLMPrompt] = []
    for role_id, grp in skills_df.groupby("role_id", sort=True):
        top = grp.sort_values("tfidf_weight", ascending=False).head(_TOP_DEFINING_SKILLS)
        skills_lines = "\n".join(
            f"- {row['skill_canonical_name']} (TF-IDF {row['tfidf_weight']:.4f})"
            for _, row in top.iterrows()
        )
        roles.append(int(role_id))
        prompts.append(LLMPrompt(
            system=_LLM_SOFT_SYSTEM_PROMPT,
            user=(
                f"Роль (role_id={int(role_id)}). Ключевые навыки:\n{skills_lines}\n\n"
                "Мягкие компетенции для оценки:\n"
                f"{definitions}\n\n"
                "Оцени каждую компетенцию по шкале 1–5 и верни JSON со списком оценок."
            ),
        ))

    if not prompts:
        return {
            role_id: _apply_rubric(hard, rubric, soft_ids)
            for role_id, hard in role_hard.items()
        }

    client = build_llm(config)
    print(f"[Phase 5b] Scoring {len(prompts)} roles via LLM ({config.llm_model})")
    results = asyncio.run(client.run_many(prompts, SoftScoresResponse))

    scored: dict[int, dict[str, float]] = {}
    for role_id, result in zip(roles, results, strict=True):
        rubric_soft = _apply_rubric(role_hard.get(role_id, {}), rubric, soft_ids)
        if result is None:
            scored[role_id] = rubric_soft
            continue
        llm_scores = {
            score.soft_id: score.proficiency
            for score in result.scores
            if score.soft_id in soft_ids
        }
        scored[role_id] = {
            soft_id: llm_scores.get(soft_id, rubric_soft[soft_id])
            for soft_id in soft_ids
        }
    return scored


def score_soft_competences(config: Config) -> pd.DataFrame:
    """Derive soft competences from hard-axis scores via rubric or the LLM.

    Reads ``characteristic_scores.parquet`` and derives each role's four soft
    axes, dispatching on ``config.use_llm_phase_5b``:

    * ``False`` (default) — the fixed ``soft_rubric`` matrix is applied to each
      role's six content hard-axis scores
      (``soft = Σ_hard rubric[soft][hard] * hard_score``).  Because each rubric
      row sums to 1.0, the result is already within [1, 5].
    * ``True`` — one LLM call per role over the shared :class:`LLMClient`,
      scoring the four soft axes directly from the role's top-20 defining
      skills. A role whose LLM call totally fails (``None``) falls back to the
      rubric result for that role only.

    Both paths clip to [1, 5] and round to 4 decimals.

    Args:
        config: Pipeline configuration. ``characteristic_scores.parquet`` must
            exist (run :func:`krm.phase_5_axes.map_to_characteristics` first).

    Returns:
        DataFrame with columns ``role_id`` (int), ``soft_id`` (str),
        ``proficiency`` (float) — one row per (role × 4 soft axes).
    """
    characteristic_scores = read_parquet(config.characteristic_scores_path)
    soft_ids = config.soft_competence_ids
    rubric = config.soft_rubric

    empty_columns = {
        "role_id": pd.Series([], dtype="int64"),
        "soft_id": pd.Series([], dtype="object"),
        "proficiency": pd.Series([], dtype="float64"),
    }
    if characteristic_scores.empty:
        empty = pd.DataFrame(empty_columns)
        write_parquet(empty, config.soft_scores_path)
        return empty

    role_hard: dict[int, dict[str, float]] = {}
    for role_id, grp in characteristic_scores.groupby("role_id", sort=True):
        role_hard[int(role_id)] = {
            str(row["characteristic_id"]): float(row["proficiency"])
            for _, row in grp.iterrows()
        }

    if config.use_llm_phase_5b:
        soft_by_role = _score_soft_llm(config, role_hard)
    else:
        soft_by_role = {
            role_id: _apply_rubric(hard, rubric, soft_ids)
            for role_id, hard in role_hard.items()
        }

    rows: list[dict[str, object]] = []
    for role_id in sorted(soft_by_role):
        soft = soft_by_role[role_id]
        for soft_id in soft_ids:
            rows.append({
                "role_id": role_id,
                "soft_id": soft_id,
                "proficiency": round(
                    min(_PROF_MAX, max(_PROF_MIN, soft[soft_id])), 4
                ),
            })

    out = pd.DataFrame(rows, columns=["role_id", "soft_id", "proficiency"])
    write_parquet(out, config.soft_scores_path)
    print(f"[Phase 5b] Wrote {len(out)} soft-competence scores to {config.soft_scores_path}")
    return out
