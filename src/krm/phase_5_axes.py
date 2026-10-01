"""Phase 5: Taxonomy-driven skill-to-characteristic mapping.

Maps each role's skills onto all seven content competency axes (including
``t_profile`` / Кругозор) as ``0.7 * tag + 0.3 * cosine``, where ``tag`` is a
curated skill→axis taxonomy membership (case-insensitive glossary match, the
authoritative grouping) and ``cosine`` is similarity to the axis's
seed-glossary centroid (``deepvk/USER-bge-m3`` embeddings).  All seven axes are
globally normalized across roles (never within-role) to a 1–5 proficiency
scale.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel

from krm.config import Config
from krm.lib.embeddings import Embedder
from krm.lib.io import read_parquet, write_parquet
from krm.lib.llm import LLMClient, LLMPrompt, build_llm

_PROF_MIN = 1.0
_PROF_MAX = 5.0
_TAG_WEIGHT = 0.7
_COS_WEIGHT = 0.3
_MIN_ROLES_FOR_ZSCORE = 15
_TOP_CONTRIBUTING = 3


class AxisTagResponse(BaseModel):
    """LLM classification of a single skill lemma into one content axis."""

    axis: str


def _global_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize each column across roles to [1, 5] (z-score or global min-max)."""
    n_roles = matrix.shape[0]
    out = np.empty_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if n_roles >= _MIN_ROLES_FOR_ZSCORE:
            mu = float(col.mean())
            sigma = float(col.std())
            if sigma > 1e-12:
                out[:, j] = np.clip(3.0 + (col - mu) / sigma, _PROF_MIN, _PROF_MAX)
            else:
                out[:, j] = 3.0
        else:
            lo = float(col.min())
            span = float(col.max()) - lo
            if span > 1e-12:
                out[:, j] = _PROF_MIN + (_PROF_MAX - _PROF_MIN) * (col - lo) / span
            else:
                out[:, j] = 3.0
    return out


def _axis_centroids(
    axes: list[str],
    glossaries: dict[str, list[str]],
    embedder: Embedder,
) -> dict[str, np.ndarray]:
    """Compute an L2-normalized centroid embedding per content axis."""
    centroids: dict[str, np.ndarray] = {}
    for axis in axes:
        phrases = glossaries.get(axis, [])
        if not phrases:
            continue
        centroid = embedder.encode(phrases).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-12:
            centroid = centroid / norm
        centroids[axis] = centroid
    return centroids


def _skill_tag_matrix(
    skills: list[str],
    axes: list[str],
    glossaries: dict[str, list[str]],
) -> np.ndarray:
    """Build the curated skill→axis taxonomy tag matrix.

    A skill is tagged to an axis iff it appears (case-insensitive exact match)
    in that axis's seed glossary.  Skills appearing in several glossaries
    normalize their tag mass across the tagged axes so the row sums to 1.
    """
    tags = np.zeros((len(skills), len(axes)), dtype=float)
    axis_to_idx = {axis: j for j, axis in enumerate(axes)}
    lower_glossaries = {
        axis: {phrase.strip().lower() for phrase in glossaries.get(axis, [])}
        for axis in axes
    }
    for i, skill in enumerate(skills):
        key = skill.strip().lower()
        hit_axes = [axis for axis in axes if key in lower_glossaries[axis]]
        if hit_axes:
            for axis in hit_axes:
                tags[i, axis_to_idx[axis]] = 1.0 / len(hit_axes)
    return tags


def _axis_tag_prompts(
    skills: list[str],
    axes: list[str],
    labels: dict[str, str],
) -> list[LLMPrompt]:
    """Build one classification prompt per skill, enumerating the candidate axes."""
    axis_lines = [f"- {axis}: {labels.get(axis, axis)}" for axis in axes]
    system = (
        "Ты — эксперт по классификации профессиональных навыков научных "
        "сотрудников. Отнеси каждый навык ровно к одной оси компетенций из "
        "списка ниже и верни её id в поле axis.\n"
        "Оси компетенций:\n" + "\n".join(axis_lines)
    )
    return [LLMPrompt(system=system, user=f"Навык: {skill}") for skill in skills]


def _llm_axis_tags(
    client: LLMClient,
    skills: list[str],
    axes: list[str],
    labels: dict[str, str],
) -> dict[str, str | None]:
    """Classify each skill into one axis via the LLM, mapping failures to ``None``."""
    prompts = _axis_tag_prompts(skills, axes, labels)
    results = asyncio.run(client.run_many(prompts, AxisTagResponse))
    valid_axes = set(axes)
    return {
        skill: (res.axis if res is not None and res.axis in valid_axes else None)
        for skill, res in zip(skills, results, strict=True)
    }


def _merge_llm_axis_tags(
    config: Config,
    skills: list[str],
    axes: list[str],
    tag_matrix: np.ndarray,
) -> np.ndarray:
    """Fill glossary-untagged skill rows with LLM axis tags.

    Only rows whose curated-glossary membership set is empty (an all-zero tag
    row, i.e. ``hit_axes`` empty in :func:`_skill_tag_matrix`) are candidates.
    Already-tagged skills keep their deterministic tag row untouched; an LLM
    ``None`` result leaves the row untagged so the cosine term is the only
    signal, exactly as the deterministic path would score it.
    """
    if not skills or not axes:
        return tag_matrix

    untagged = [i for i, row in enumerate(tag_matrix) if float(row.sum()) == 0.0]
    if not untagged:
        return tag_matrix

    labels = config.characteristic_labels_ru
    client = build_llm(config)
    untagged_skills = [skills[i] for i in untagged]
    axis_by_skill = _llm_axis_tags(client, untagged_skills, axes, labels)

    merged = tag_matrix.copy()
    axis_to_idx = {axis: j for j, axis in enumerate(axes)}
    for i, skill in zip(untagged, untagged_skills, strict=True):
        axis = axis_by_skill.get(skill)
        if axis is not None and axis in axis_to_idx:
            merged[i, axis_to_idx[axis]] = 1.0
    return merged


def _top_contributing(
    skill_names: list[str],
    axis_scores: np.ndarray,
    top_k: int = _TOP_CONTRIBUTING,
) -> list[dict[str, float]]:
    """Return the top-*k* skills for one axis, ranked by blended score."""
    order = np.argsort(-axis_scores)[:top_k]
    return [
        {"skill": skill_names[i], "contribution": round(float(axis_scores[i]), 4)}
        for i in order
    ]


def _empty_characteristic_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "role_id": pd.Series([], dtype="int64"),
        "characteristic_id": pd.Series([], dtype="object"),
        "proficiency": pd.Series([], dtype="float64"),
        "top_contributing_skills": pd.Series([], dtype="object"),
    })


def map_to_characteristics(config: Config) -> pd.DataFrame:
    """Map skills to competency characteristics via curated tags + embeddings.

    Reads ``skills_per_role.parquet``, scores each skill against each of the 7
    content axes as ``0.7 * tag + 0.3 * cosine``, aggregates per-role means,
    globally normalizes to [1, 5], and writes ``characteristic_scores.parquet``
    and the skill×axis matrix ``skill_characteristic_scores.parquet``.

    When ``config.use_llm_phase_5`` is true, skills the curated glossary cannot
    tag (empty ``hit_axes``) are classified into one axis by the shared LLM
    client and merged into the tag matrix before the cosine blend. The 0.7/0.3
    weights, the cosine blend, and global normalization are unchanged.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame with one row per role–characteristic pair (7 axes per role).

    Raises:
        FileNotFoundError: If ``skills_per_role.parquet`` does not exist.
    """
    skills_df = read_parquet(config.output_dir / "skills_per_role.parquet")
    characteristic_ids = config.characteristic_ids
    content_axes = list(characteristic_ids)
    glossaries = config.characteristic_seed_glossaries

    unique_skills = sorted(skills_df["skill_canonical_name"].dropna().unique().tolist())
    embedder = Embedder(model_name=config.embedding_model)
    skill_emb_map = dict(zip(unique_skills, embedder.encode(unique_skills), strict=True))
    centroids = _axis_centroids(content_axes, glossaries, embedder)
    active_axes = [cid for cid in content_axes if cid in centroids]

    tag_matrix = _skill_tag_matrix(unique_skills, active_axes, glossaries)
    if config.use_llm_phase_5:
        tag_matrix = _merge_llm_axis_tags(config, unique_skills, active_axes, tag_matrix)
    if unique_skills and active_axes:
        skill_matrix = np.stack([skill_emb_map[s] for s in unique_skills])
        centroid_matrix = np.stack([centroids[c] for c in active_axes])
        cosine_matrix = skill_matrix @ centroid_matrix.T
    else:
        cosine_matrix = np.zeros((len(unique_skills), len(active_axes)), dtype=float)
    skill_axis_scores = _TAG_WEIGHT * tag_matrix + _COS_WEIGHT * cosine_matrix

    role_ids = sorted(skills_df["role_id"].unique().tolist())
    raw_matrix = np.zeros((len(role_ids), len(active_axes)), dtype=float)
    role_skill_indices: dict[int, list[int]] = {}
    skill_to_idx = {skill: i for i, skill in enumerate(unique_skills)}
    for r, role_id in enumerate(role_ids):
        names = skills_df.loc[skills_df["role_id"] == role_id, "skill_canonical_name"].tolist()
        indices = [skill_to_idx[n] for n in names if n in skill_to_idx]
        role_skill_indices[role_id] = indices
        if indices:
            raw_matrix[r] = skill_axis_scores[indices].mean(axis=0)

    norm_matrix = _global_normalize(raw_matrix) if raw_matrix.size else raw_matrix

    records: list[dict[str, object]] = []
    for r, role_id in enumerate(role_ids):
        indices = role_skill_indices[role_id]
        names = [unique_skills[i] for i in indices]
        role_scores = skill_axis_scores[indices] if indices else np.zeros((0, len(active_axes)))
        for j, axis in enumerate(active_axes):
            top = _top_contributing(names, role_scores[:, j]) if indices else []
            records.append({
                "role_id": int(role_id),
                "characteristic_id": axis,
                "proficiency": round(float(norm_matrix[r, j]), 4),
                "top_contributing_skills": json.dumps(top, ensure_ascii=False),
            })

    characteristic_df = _empty_characteristic_frame() if not records else pd.DataFrame(
        records, columns=[
            "role_id", "characteristic_id", "proficiency", "top_contributing_skills",
        ]
    )
    write_parquet(characteristic_df, config.characteristic_scores_path)

    matrix_rows = []
    for i, skill in enumerate(unique_skills):
        for j, axis in enumerate(active_axes):
            matrix_rows.append({
                "skill_canonical_name": skill,
                "characteristic_id": axis,
                "nli_score": round(float(skill_axis_scores[i, j]), 4),
            })
    matrix_df = pd.DataFrame(
        matrix_rows, columns=["skill_canonical_name", "characteristic_id", "nli_score"]
    )
    write_parquet(matrix_df, config.skill_characteristic_scores_path)

    print(
        f"[Phase 5] Wrote {len(characteristic_df)} role–characteristic scores "
        f"to {config.characteristic_scores_path}"
    )
    return characteristic_df
