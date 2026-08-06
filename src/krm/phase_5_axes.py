"""Phase 5: Zero-Shot NLI Skill-to-Axis Mapping.

Maps extracted skills onto competency axes using zero-shot natural language
inference (NLI). Each skill is scored against six axis hypotheses defined
in ``config.yaml``, producing a soft assignment that allows a single skill
to contribute to multiple competency axes.

Results are aggregated per role using TF-IDF weights and normalized to
a 1–5 proficiency scale.

Usage:
    from krm.config import Config
    from krm.phase_5_axes import map_to_axes

    config = Config()
    axis_df = map_to_axes(config)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from krm.config import Config
from krm.lib.io import read_parquet, write_parquet


def _build_pipeline(config: Config) -> Any:
    """Initialise the zero-shot classification pipeline.

    Reuses the model configured for Phase 2 classification
    (``classification.model``, default ``facebook/bart-large-mnli``).

    Args:
        config: Pipeline configuration.

    Returns:
        A HuggingFace ``zero-shot-classification`` pipeline.
    """
    from transformers import pipeline

    device: str = config._data.get("classification", {}).get("device", "cpu")
    return pipeline(
        "zero-shot-classification",
        model=config.classification_model,
        device=device,
    )


def _score_skills(
    unique_skills: list[str],
    hypotheses: list[str],
    classifier: Any,
    batch_size: int = 32,
) -> dict[str, dict[str, float]]:
    """Score every unique skill against all axis hypotheses using zero-shot NLI.

    Results are cached per skill so that a skill appearing in multiple
    roles is only processed once.

    Args:
        unique_skills: Deduplicated list of skill strings.
        hypotheses: Axis hypothesis texts from ``config.axis_hypotheses``.
        classifier: A HuggingFace zero-shot-classification pipeline.
        batch_size: Number of texts to process per classifier call.

    Returns:
        Nested dict ``{skill: {hypothesis: entailment_score}}``.
    """
    cache: dict[str, dict[str, float]] = {}

    for batch_start in range(0, len(unique_skills), batch_size):
        batch_end = min(batch_start + batch_size, len(unique_skills))
        batch = unique_skills[batch_start:batch_end]

        results: list[dict[str, Any]] = classifier(batch, hypotheses)

        for skill, result in zip(batch, results):
            cache[skill] = dict(zip(result["labels"], result["scores"]))

    return cache


def _normalize_to_proficiency(
    raw_scores: dict[str, float],
    min_val: float = 1.0,
    max_val: float = 5.0,
) -> dict[str, float]:
    """Normalise a dict of ``{axis_label: score}`` to the proficiency scale.

    Min-max normalisation is applied *within* the values of the dict
    (i.e. across all axes for a single role), then linearly scaled to
    [``min_val``, ``max_val``].

    If all scores are identical (zero range), every axis receives the
    midpoint of the scale.

    Args:
        raw_scores: Mapping from axis label to raw aggregated score.
        min_val: Lower bound of the target scale (default 1.0).
        max_val: Upper bound of the target scale (default 5.0).

    Returns:
        Normalised ``{axis_label: proficiency}`` mapping.
    """
    values = list(raw_scores.values())
    v_min = min(values)
    v_max = max(values)
    span = v_max - v_min

    if span == 0.0:
        if v_min < 0.001:
            return {k: min_val for k in raw_scores}
        midpoint = (min_val + max_val) / 2.0
        return {k: midpoint for k in raw_scores}

    scale = (max_val - min_val) / span
    return {k: min_val + (v - v_min) * scale for k, v in raw_scores.items()}


def _top_contributing(
    role_skills: pd.DataFrame,
    nli_cache: dict[str, dict[str, float]],
    axis_id: str,
    hypothesis: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return the top-*k* contributing skills for a single axis within a role.

    Skills are ranked by ``tfidf_weight * nli_score[hypothesis]``.

    Args:
        role_skills: DataFrame slice for a single role (columns must include
            ``skill`` and ``tfidf_weight``).
        nli_cache: Pre-computed NLI scores per skill.
        axis_id: Axis identifier (e.g. ``"experimental"``).
        hypothesis: The hypothesis text used as the NLI label.
        top_k: Number of top skills to return (default 3).

    Returns:
        List of ``{"skill": str, "contribution": float}`` dicts, sorted
        by contribution descending.
    """
    contributions: list[dict[str, Any]] = []

    for _, row in role_skills.iterrows():
        skill: str = row["skill"]
        weight: float = float(row["tfidf_weight"])
        nli_score = nli_cache.get(skill, {}).get(hypothesis, 0.0)
        contrib = weight * nli_score
        contributions.append({"skill": skill, "contribution": float(contrib)})

    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    return contributions[:top_k]


def map_to_axes(config: Config) -> pd.DataFrame:
    """Map skills to competency axes using zero-shot NLI and aggregate per role.

    Full pipeline:

    1. Reads ``skills_per_role.parquet`` from ``config.output_dir``.
    2. Collects all unique skill strings across roles.
    3. Loads ``facebook/bart-large-mnli`` via the HuggingFace
       zero-shot-classification pipeline.
    4. Scores every unique skill against the six axis hypotheses
       defined in ``config.axes`` (results cached globally — skills
       shared across roles are not re-processed).
    5. For each role, aggregates axis scores as the TF-IDF-weighted
       mean of per-skill NLI scores:
       ``proficiency[axis] = Σ(tfidf_w * nli_score[axis]) / Σ(tfidf_w)``.
    6. Normalises aggregated scores to a 1–5 proficiency scale via
       min-max scaling within the role's axes.
    7. Identifies the top-3 contributing skills per axis (by weighted
       contribution) and stores them as JSON.
    8. Writes ``axis_scores.parquet`` to ``config.output_dir``.

    Args:
        config: Pipeline configuration providing ``axis_hypotheses``,
            ``axis_ids``, and ``n_axes``.

    Returns:
        DataFrame with one row per role–axis pair:

        ========================== ==========================================
        column                     description
        ========================== ==========================================
        ``role_id``                int — role identifier
        ``axis_id``                str — axis code (e.g. ``"experimental"``)
        ``proficiency``            float — normalised 1–5 proficiency score
        ``top_contributing_skills`` str — JSON array of ``[{skill, contribution}]``
        ========================== ==========================================

    Raises:
        FileNotFoundError: If ``skills_per_role.parquet`` does not exist.
    """
    # ------------------------------------------------------------------
    # 1. Load skills per role
    # ------------------------------------------------------------------
    skills_path = config.output_dir / "skills_per_role.parquet"
    skills_df = read_parquet(skills_path)
    print(
        f"[Phase 5] Loaded {len(skills_df)} skill–role assignments "
        f"from {skills_path}"
    )

    # ------------------------------------------------------------------
    # 2. Collect unique skills
    # ------------------------------------------------------------------
    unique_skills: list[str] = sorted(skills_df["skill_canonical_name"].dropna().unique().tolist())
    print(f"[Phase 5] {len(unique_skills)} unique skills to classify")

    # ------------------------------------------------------------------
    # 3. Load NLI classifier
    # ------------------------------------------------------------------
    hypotheses: list[str] = config.axis_hypotheses
    axis_ids: list[str] = config.axis_ids
    print(f"[Phase 5] Loading NLI model: {config.classification_model}")
    classifier = _build_pipeline(config)
    print(f"[Phase 5] {len(hypotheses)} axis hypotheses loaded")

    # ------------------------------------------------------------------
    # 4. Score all unique skills (with global cache)
    # ------------------------------------------------------------------
    nli_cache = _score_skills(unique_skills, hypotheses, classifier)

    # ------------------------------------------------------------------
    # 5. Aggregate per role
    # ------------------------------------------------------------------
    role_ids_sorted: list[int] = sorted(skills_df["role_id"].unique().tolist())
    axis_records: list[dict[str, Any]] = []

    for role_id in role_ids_sorted:
        role_mask = skills_df["role_id"] == role_id
        role_skills = skills_df[role_mask]

        # --- Weighted sum for each axis ---
        raw_axis_scores: dict[str, float] = {}
        total_weight: float = float(role_skills["tfidf_weight"].sum())

        if total_weight == 0.0:
            # Edge case: no meaningful weights — assign midpoint.
            for axis_id, hypothesis in zip(axis_ids, hypotheses):
                axis_records.append({
                    "role_id": int(role_id),
                    "axis_id": axis_id,
                    "proficiency": 3.0,
                    "top_contributing_skills": json.dumps([], ensure_ascii=False),
                })
            continue

        for axis_id, hypothesis in zip(axis_ids, hypotheses):
            weighted_sum: float = 0.0
            for _, row in role_skills.iterrows():
                skill: str = row["skill_canonical_name"]
                weight: float = float(row["tfidf_weight"])
                nli_score = nli_cache.get(skill, {}).get(hypothesis, 0.0)
                weighted_sum += weight * nli_score
            raw_axis_scores[axis_id] = weighted_sum / total_weight

        # --- Normalise to 1–5 scale ---
        norm_scores = _normalize_to_proficiency(raw_axis_scores)

        # --- Top contributing skills per axis ---
        for axis_id, hypothesis in zip(axis_ids, hypotheses):
            top_skills = _top_contributing(
                role_skills,
                nli_cache,
                axis_id,
                hypothesis,
                top_k=3,
            )
            axis_records.append({
                "role_id": int(role_id),
                "axis_id": axis_id,
                "proficiency": round(norm_scores[axis_id], 4),
                "top_contributing_skills": json.dumps(
                    top_skills, ensure_ascii=False
                ),
            })

    # ------------------------------------------------------------------
    # 6. Build output DataFrame
    # ------------------------------------------------------------------
    axis_df = pd.DataFrame(axis_records)
    # Ensure consistent column ordering.
    axis_df = axis_df[[
        "role_id",
        "axis_id",
        "proficiency",
        "top_contributing_skills",
    ]]

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    output_path = config.output_dir / "axis_scores.parquet"
    write_parquet(axis_df, output_path)
    print(
        f"[Phase 5] Wrote {len(axis_df)} role–axis scores to {output_path}"
    )

    return axis_df
