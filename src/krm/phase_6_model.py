"""Phase 6: Competency Role Models and Spider Chart Visualization.

Builds structured competency models for each discovered role using
characteristic proficiency scores and generates spider (radar) chart PNGs.

Usage:
    from krm.config import Config
    from krm.phase_6_model import build_models

    config = Config()
    models = build_models(config)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from krm.config import Config
from krm.lib.io import read_parquet
from krm.visualization import (
    compute_experience_bins,
    render_role_experience,
    render_role_skills,
    render_role_soft_spider,
    render_role_spider,
)

# ---------------------------------------------------------------------------
# 7-level competency scale: бакалавр → главный научный сотрудник
# ---------------------------------------------------------------------------

_COMPETENCY_LEVELS: list[dict[str, Any]] = [
    {"id": 0, "label": "Бакалавр", "weight": 0.10},
    {"id": 1, "label": "Магистр", "weight": 0.25},
    {"id": 2, "label": "Аспирант / МНС", "weight": 0.40},
    {"id": 3, "label": "Научный сотрудник", "weight": 0.60},
    {"id": 4, "label": "Старший научный сотрудник", "weight": 0.80},
    {"id": 5, "label": "Ведущий научный сотрудник", "weight": 0.95},
    {"id": 6, "label": "Главный научный сотрудник", "weight": 1.00},
]

_PROFICIENCY_MIN = 1.0
_PROFICIENCY_MAX = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weight_to_threshold(weight: float) -> float:
    """Map a competency level weight (0–1) onto the 1–5 proficiency scale."""
    return _PROFICIENCY_MIN + (_PROFICIENCY_MAX - _PROFICIENCY_MIN) * weight


def _parse_json_field(value: Any) -> list[Any]:
    """Parse a JSON‑string column or pass through a list unchanged."""
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, list):
        return list(value)
    return []


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_single_model(
    role_row: pd.Series,
    characteristic_rows: pd.DataFrame,
    characteristic_label_map: dict[str, str],
) -> dict[str, Any]:
    """Assemble a competency model dict for one role.

    Args:
        role_row: Single row from ``roles.parquet`` (noise already excluded).
        characteristic_rows: All characteristic-score rows for this role.
        characteristic_label_map: Mapping ``characteristic_id → label_ru``
            from config.

    Returns:
        Model dict with keys: ``role_id``, ``role_label``, ``top_job_titles``,
        ``member_count``, ``characteristics``, ``competency_levels``.
    """
    role_id = int(role_row["role_id"])

    # -- Role identity --
    model: dict[str, Any] = {
        "role_id": role_id,
        "role_label": str(role_row["role_label"]),
        "top_job_titles": _parse_json_field(role_row["top_titles"]),
        "member_count": int(role_row["member_count"]),
    }

    # -- Characteristic proficiency profile --
    characteristics: list[dict[str, Any]] = []
    characteristic_proficiencies: dict[str, float] = {}

    for _, ar in characteristic_rows.iterrows():
        characteristic_id = str(ar["characteristic_id"])
        proficiency = float(ar["proficiency"])
        top_skills = _parse_json_field(ar["top_contributing_skills"])[:3]

        characteristics.append(
            {
                "characteristic_id": characteristic_id,
                "label_ru": characteristic_label_map.get(characteristic_id, characteristic_id),
                "proficiency": proficiency,
                "top_skills": top_skills,
            }
        )
        characteristic_proficiencies[characteristic_id] = proficiency

    model["characteristics"] = characteristics

    # -- 7-level scale with per-characteristic achievement --
    competency_levels: list[dict[str, Any]] = []
    for level in _COMPETENCY_LEVELS:
        threshold = _weight_to_threshold(level["weight"])
        characteristics_achieved = [
            char_id
            for char_id, prof in characteristic_proficiencies.items()
            if prof >= threshold
        ]
        competency_levels.append(
            {
                "label": level["label"],
                "weight": level["weight"],
                "threshold": threshold,
                "characteristics_achieved": characteristics_achieved,
                "achieved_count": len(characteristics_achieved),
            }
        )

    model["competency_levels"] = competency_levels

    return model


# ---------------------------------------------------------------------------
# Diagrams (matplotlib, no plotly)
# ---------------------------------------------------------------------------


def _generate_role_diagrams(
    model: dict[str, Any],
    experience_years: list[float],
    config: Config,
) -> None:
    """Render the 4 per-role diagrams into ``reports_dir/<kind>/{role_id}.png``."""
    role_id = model["role_id"]

    hard_dir = config.reports_dir / "hard"
    hard_dir.mkdir(parents=True, exist_ok=True)
    render_role_spider(model, config.characteristic_ids, hard_dir / f"{role_id}.png")

    soft_dir = config.reports_dir / "soft"
    soft_dir.mkdir(parents=True, exist_ok=True)
    render_role_soft_spider(model, config.soft_competence_ids, soft_dir / f"{role_id}.png")

    experience_dir = config.reports_dir / "experience"
    experience_dir.mkdir(parents=True, exist_ok=True)
    render_role_experience(
        model["role_label"],
        experience_years,
        experience_dir / f"{role_id}.png",
        bins=config.experience_bins,
    )

    skills_dir = config.reports_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    render_role_skills(
        model["role_label"],
        model["skills"],
        config.characteristic_labels_ru,
        skills_dir / f"{role_id}.png",
        top_n=config.skills_top_n,
    )


# ---------------------------------------------------------------------------
# Auxiliary inputs (soft scores, archetypes, experience, skills)
# ---------------------------------------------------------------------------


def _read_archetype_map(config: Config) -> dict[int, str]:
    df = read_parquet(config.role_archetypes_path)
    return {int(r["role_id"]): str(r["archetype"]) for _, r in df.iterrows()}


def _read_soft_scores(config: Config) -> dict[int, dict[str, float]]:
    df = read_parquet(config.soft_scores_path)
    result: dict[int, dict[str, float]] = {}
    for _, r in df.iterrows():
        result.setdefault(int(r["role_id"]), {})[str(r["soft_id"])] = float(r["proficiency"])
    return result


def _read_experience_years(config: Config) -> dict[int, list[float]]:
    vacancy_years = read_parquet(config.vacancy_experience_path)
    vacancy_roles = read_parquet(config.vacancy_roles_path)
    merged = vacancy_years.merge(vacancy_roles, on="vacancy_id", how="inner")
    result: dict[int, list[float]] = {}
    for role_id, grp in merged.groupby("role_id"):
        result[int(role_id)] = [float(y) for y in grp["experience_years"].tolist()]
    return result


def _read_skill_axis_weights(config: Config) -> dict[str, dict[str, float]]:
    df = read_parquet(config.skill_characteristic_scores_path)
    result: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        skill = str(r["skill_canonical_name"])
        result.setdefault(skill, {})[str(r["characteristic_id"])] = float(r["nli_score"])
    return result


# ---------------------------------------------------------------------------
# Extended model blocks
# ---------------------------------------------------------------------------


def _build_soft_competences(
    role_id: int,
    soft_scores: dict[int, dict[str, float]],
    config: Config,
) -> list[dict[str, Any]]:
    labels = config.soft_competence_labels_ru
    role_soft = soft_scores.get(role_id, {})
    return [
        {
            "soft_id": sid,
            "label_ru": labels.get(sid, sid),
            "proficiency": role_soft.get(sid, 3.0),
        }
        for sid in config.soft_competence_ids
    ]


def _build_experience(years: list[float], config: Config) -> dict[str, Any]:
    labelled, median, n_not_specified = compute_experience_bins(years, config.experience_bins)
    return {
        "bins": [{"range": rng, "count": cnt} for rng, cnt in labelled],
        "median": median,
        "n_with_experience": len(years) - n_not_specified,
        "n_not_specified": n_not_specified,
    }


def _build_skills(
    role_id: int,
    skills_df: pd.DataFrame,
    axis_weights: dict[str, dict[str, float]],
    config: Config,
) -> list[dict[str, Any]]:
    role_skills = skills_df[skills_df["role_id"] == role_id]
    top = role_skills.sort_values("tfidf_weight", ascending=False).head(config.skills_top_n)
    return [
        {
            "skill": str(r["skill_canonical_name"]),
            "tfidf_weight": float(r["tfidf_weight"]),
            "axis_weights": axis_weights.get(str(r["skill_canonical_name"]), {}),
        }
        for _, r in top.iterrows()
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_models(config: Config) -> list[dict[str, Any]]:
    """Build extended competency role models and their 4 diagrams.

    Reads ``roles.parquet``, ``characteristic_scores.parquet`` and the five
    auxiliary inputs (``role_archetypes``, ``soft_scores``, ``vacancy_roles``,
    ``characteristics``, ``skills_per_role``, ``skill_characteristic_scores``).
    For each non‑noise role the model dict gains ``archetype``,
    ``soft_competences``, ``experience`` and ``skills`` on top of the existing
    ``characteristics`` (7 hard axes) and ``competency_levels``.

    Writes ``{config.models_dir}/{role_id}.json`` and renders 4 diagrams to
    ``{config.reports_dir}/{hard|soft|experience|skills}/{role_id}.png``.

    Args:
        config: Pipeline configuration providing the input paths, axis ids,
            labels, and output directories.

    Returns:
        List of extended model dicts, one per non‑noise role.

    Raises:
        FileNotFoundError: If any required input Parquet file is missing.
    """
    roles_df = read_parquet(config.roles_path)
    characteristic_scores_df = read_parquet(config.characteristic_scores_path)

    roles_clean = roles_df[~roles_df["noise_flag"]]

    if len(roles_clean) == 0:
        print("[Phase 6] No non-noise roles — nothing to model")
        return []

    characteristic_label_map = config.characteristic_labels_ru

    archetype_map = _read_archetype_map(config)
    soft_scores = _read_soft_scores(config)
    experience_map = _read_experience_years(config)
    skills_df = read_parquet(config.skills_per_role_path)
    axis_weights = _read_skill_axis_weights(config)

    config.models_dir.mkdir(parents=True, exist_ok=True)

    models: list[dict[str, Any]] = []

    for _, role_row in roles_clean.iterrows():
        role_id = int(role_row["role_id"])

        characteristic_rows = characteristic_scores_df[
            characteristic_scores_df["role_id"] == role_id
        ]
        if len(characteristic_rows) == 0:
            print(
                f"  [Phase 6] Warning: no characteristic scores for role {role_id}, skipping"
            )
            continue

        model = _build_single_model(role_row, characteristic_rows, characteristic_label_map)

        model["archetype"] = archetype_map.get(role_id, "")
        model["soft_competences"] = _build_soft_competences(role_id, soft_scores, config)
        years = experience_map.get(role_id, [])
        model["experience"] = _build_experience(years, config)
        model["skills"] = _build_skills(role_id, skills_df, axis_weights, config)

        json_path = config.models_dir / f"{role_id}.json"
        json_path.write_text(
            json.dumps(model, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _generate_role_diagrams(model, years, config)

        models.append(model)

    print(f"[Phase 6] Built {len(models)} competency models")
    return models
