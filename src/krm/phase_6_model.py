"""Phase 6: Competency Role Models and Spider Chart Visualization.

Builds structured competency models for each discovered role using
axis proficiency scores and generates spider (radar) chart PNGs.

Usage:
    from krm.config import Config
    from krm.phase_6_model import build_models

    config = Config()
    models = build_models(config)
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from krm.config import Config
from krm.lib.io import read_parquet

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
    axis_rows: pd.DataFrame,
    axis_label_map: dict[str, str],
) -> dict[str, Any]:
    """Assemble a competency model dict for one role.

    Args:
        role_row: Single row from ``roles.parquet`` (noise already excluded).
        axis_rows: All axis‑score rows for this role.
        axis_label_map: Mapping ``axis_id → label_ru`` from config.

    Returns:
        Model dict with keys: ``role_id``, ``role_label``, ``top_job_titles``,
        ``member_count``, ``axes``, ``competency_levels``.
    """
    role_id = int(role_row["role_id"])

    # -- Role identity --
    model: dict[str, Any] = {
        "role_id": role_id,
        "role_label": str(role_row["role_label"]),
        "top_job_titles": _parse_json_field(role_row["top_titles"]),
        "member_count": int(role_row["member_count"]),
    }

    # -- Axis proficiency profile --
    axes: list[dict[str, Any]] = []
    axis_proficiencies: dict[str, float] = {}

    for _, ar in axis_rows.iterrows():
        axis_id = str(ar["axis_id"])
        proficiency = float(ar["proficiency"])
        top_skills = _parse_json_field(ar["top_contributing_skills"])[:3]

        axes.append(
            {
                "axis_id": axis_id,
                "label_ru": axis_label_map.get(axis_id, axis_id),
                "proficiency": proficiency,
                "top_skills": top_skills,
            }
        )
        axis_proficiencies[axis_id] = proficiency

    model["axes"] = axes

    # -- 7‑level scale with per‑axis achievement --
    competency_levels: list[dict[str, Any]] = []
    for level in _COMPETENCY_LEVELS:
        threshold = _weight_to_threshold(level["weight"])
        axes_achieved = [
            ax_id for ax_id, prof in axis_proficiencies.items() if prof >= threshold
        ]
        competency_levels.append(
            {
                "label": level["label"],
                "weight": level["weight"],
                "threshold": threshold,
                "axes_achieved": axes_achieved,
                "achieved_count": len(axes_achieved),
            }
        )

    model["competency_levels"] = competency_levels

    return model


# ---------------------------------------------------------------------------
# Spider chart (matplotlib, no plotly)
# ---------------------------------------------------------------------------


def _generate_spider_chart(
    model: dict[str, Any],
    axis_order: list[str],
    config: Config,
) -> None:
    """Render a spider (radar) chart for a single role model as a PNG.

    Uses the ``axis_order`` list to ensure consistent angular placement
    (matching ``config.axis_ids`` order).  The chart is saved at 300 dpi
    to ``config.reports_dir / "spider_charts" / {role_id}.png``.

    Args:
        model: Single competency model dict (as returned by
            :func:`_build_single_model`).
        axis_order: Axis IDs in the desired clockwise order.
        config: Pipeline configuration (provides ``reports_dir``).
    """
    # Build a lookup for fast axis → record access.
    axes_by_id: dict[str, dict[str, Any]] = {ax["axis_id"]: ax for ax in model["axes"]}

    labels: list[str] = []
    values: list[float] = []
    for axis_id in axis_order:
        entry = axes_by_id.get(axis_id)
        if entry is not None:
            labels.append(entry["label_ru"])
            values.append(entry["proficiency"])

    n = len(labels)
    if n == 0:
        return

    # Close the polygon.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    # Create polar plot.
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#1f77b4")
    ax.fill(angles_plot, values_plot, alpha=0.25, color="#1f77b4")

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title(model["role_label"], fontsize=13, fontweight="bold", pad=22)
    ax.grid(True, alpha=0.3)

    output_dir = config.reports_dir / "spider_charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model['role_id']}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_models(config: Config) -> list[dict[str, Any]]:
    """Build competency role models and spider chart visualizations.

    Reads ``roles.parquet`` and ``axis_scores.parquet`` from
    ``config.output_dir``.  For each non‑noise role:

    1. Assembles a structured JSON competency model containing:
       - Role identity (label, top job titles, member count).
       - 6‑axis proficiency profile with top‑3 contributing skills per axis.
       - 7‑level competency scale (бакалавр → главный научный сотрудник)
         with per‑axis achievement thresholds.

    2. Writes the model to ``{config.models_dir}/{role_id}.json``.

    3. Renders a spider (radar) chart via matplotlib at 300 dpi to
       ``{config.reports_dir}/spider_charts/{role_id}.png``.

    Args:
        config: Pipeline configuration providing ``output_dir``,
            ``axis_ids``, ``axis_labels_ru``, ``models_dir``, and
            ``reports_dir``.

    Returns:
        List of model dicts, one per non‑noise role.  Each dict contains:

        ====================== ===========================================
        key                    description
        ====================== ===========================================
        ``role_id``            int — cluster label
        ``role_label``         str — most central job title
        ``top_job_titles``     list[str] — top central titles
        ``member_count``       int — number of unique titles
        ``axes``               list[dict] — {axis_id, label_ru,
                               proficiency (1‑5), top_skills}
        ``competency_levels``  list[dict] — {label, weight, threshold,
                               axes_achieved, achieved_count}
        ====================== ===========================================

    Raises:
        FileNotFoundError: If ``roles.parquet`` or ``axis_scores.parquet``
            is missing from ``config.output_dir``.
    """
    # ------------------------------------------------------------------
    # 1. Read inputs
    # ------------------------------------------------------------------
    roles_path = config.output_dir / "roles.parquet"
    axis_scores_path = config.output_dir / "axis_scores.parquet"

    roles_df = read_parquet(roles_path)
    axis_scores_df = read_parquet(axis_scores_path)

    # 2. Exclude noise roles.
    roles_clean = roles_df[roles_df["noise_flag"] != True]  # noqa: E712

    if len(roles_clean) == 0:
        print("[Phase 6] No non-noise roles — nothing to model")
        return []

    # 3. Pre-compute axis label lookup.
    axis_label_map = dict(zip(config.axis_ids, config.axis_labels_ru))

    # 4. Ensure output directories.
    config.models_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = config.reports_dir / "spider_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # 5. Build models.
    models: list[dict[str, Any]] = []

    for _, role_row in roles_clean.iterrows():
        role_id = int(role_row["role_id"])

        axis_rows = axis_scores_df[axis_scores_df["role_id"] == role_id]
        if len(axis_rows) == 0:
            print(f"  [Phase 6] Warning: no axis scores for role {role_id}, skipping")
            continue

        model = _build_single_model(role_row, axis_rows, axis_label_map)

        # Write JSON.
        json_path = config.models_dir / f"{role_id}.json"
        json_path.write_text(
            json.dumps(model, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Generate spider chart (axes in config order for consistency).
        _generate_spider_chart(model, config.axis_ids, config)

        models.append(model)

    print(f"[Phase 6] Built {len(models)} competency models")
    return models
