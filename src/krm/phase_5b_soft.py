"""Phase 5b: Role Archetype Classification and Soft-Competence Scoring.

Two-stage pipeline run after Phase 4 (skills) and Phase 5 (hard axes):

1. :func:`classify_role_archetypes` — classifies each discovered role into one
   of six ISCB-inspired archetypes (Техник, Исследователь, Методолог, Инженер,
   Ведущий специалист, Гибрид) from its top-20 defining skills.
2. :func:`score_soft_competences` — derives each role's four soft competences
   as a fixed linear rubric applied to its six hard-axis scores.

Usage::

    from krm.config import Config
    from krm.phase_5b_soft import classify_role_archetypes, score_soft_competences

    config = Config()
    classify_role_archetypes(config)
    soft_scores = score_soft_competences(config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from krm.analysis.archetypes import classify_archetypes
from krm.config import Config
from krm.lib.io import read_parquet, write_parquet

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


def score_soft_competences(config: Config) -> pd.DataFrame:
    """Derive soft competences from hard-axis scores via the soft rubric.

    Reads ``characteristic_scores.parquet``, applies the fixed ``soft_rubric``
    matrix to each role's six content hard-axis scores
    (``soft = Σ_hard rubric[soft][hard] * hard_score``), and clips to [1, 5].
    Because each rubric row sums to 1.0, the result is already within the
    proficiency scale; no additional normalization is applied.

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

    rows: list[dict[str, object]] = []
    for role_id in sorted(role_hard):
        soft = _apply_rubric(role_hard[role_id], rubric, soft_ids)
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
