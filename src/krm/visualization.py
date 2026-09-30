"""Publication-quality competency-role model visualizations (the "эталон").

Renders the final output of the KRM pipeline — the competency-role model — as
clean, publication-ready figures:

* ``render_role_spider`` — a 7-axis spider (radar) chart of characteristic
  proficiency with top contributing skills annotated per axis.
* ``render_role_card``   — the spider chart plus the 7-level competency
  achievement matrix (characteristic × career level), i.e. the full model.
* ``render_portfolio``   — a grid of spider charts for side-by-side role
  comparison.

All renderers accept the model dict produced by
:func:`krm.phase_6_model._build_single_model` (and read the same 7-axis /
1–5 proficiency / 7-level structure that ``config.yaml`` defines), so the same
data that drives the pipeline also drives the эталон.

The styling is deliberately publication-oriented: colorblind-safe palette,
DejaVu Sans (full Cyrillic support), 300 dpi, consistent angular layout.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, TypedDict, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.projections import PolarAxes

# ---------------------------------------------------------------------------
# Model shapes (the pipeline represents role models as plain dicts)
# ---------------------------------------------------------------------------


class CareerLevel(TypedDict):
    label: str
    axes: dict[str, float]


# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------

_PRIMARY = "#2C5F8A"  # deep steel blue — fill & line
_ACCENT = "#F4A261"  # warm amber — markers / highlights (reserved)
_GRID = "#C7D2DA"
_MUTED = "#6B7280"
_TEXT = "#1F2937"
_NOT_ACHIEVED = "#EDEFF2"
_CELL_EDGE = "#D3D9DE"

# Sequential blue ramp for the achievement matrix (light → saturated).
_ACHIEVEMENT_CMAP = LinearSegmentedColormap.from_list(
    "krm_blues", ["#D6E4F0", _PRIMARY], N=256
)

# Short labels for the achievement-matrix x-axis (full labels live on the radar).
_SHORT_LABELS: dict[str, str] = {
    "domain_knowledge": "Домен",
    "experimental": "Эксперимент",
    "data_analysis": "Анализ",
    "computational": "Вычисления",
    "professional_texts": "Тексты",
    "t_profile": "T-профиль",
    "management": "Управление",
}

_PROFICIENCY_MIN = 1.0
_PROFICIENCY_MAX = 5.0

_LABEL_WRAP = 14
_SKILL_WRAP = 22
_LABEL_R = 5.75  # radial distance of axis labels (grid ends at 5.0)
_SKILL_GAP = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skill_label(skill: object) -> str:
    """Normalise a skill entry to a string (handles both ``str`` and dict forms)."""
    if isinstance(skill, dict):
        name = skill.get("skill") or skill.get("name") or skill.get("label")
        if isinstance(name, str):
            return name
    return str(skill)


def _wrap(text: str, width: int, max_lines: int = 2) -> list[str]:
    """Wrap text to ``width`` chars, truncating to ``max_lines`` with an ellipsis."""
    if not text:
        return []
    lines = textwrap.wrap(text, width=width, break_long_words=False) or [text]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "…"
    return lines


def _ordered_characteristics(
    model: dict[str, Any], order: list[str]
) -> list[dict[str, Any]]:
    """Return model characteristics in ``order`` (skipping any missing)."""
    by_id = {c["characteristic_id"]: c for c in model["characteristics"]}
    return [by_id[cid] for cid in order if cid in by_id]


def _bin_label(lo: float, hi: float) -> str:
    """Format one histogram bin edge pair (``hi=inf`` renders the open-ended bin)."""
    if np.isinf(hi):
        return f"{lo:g}+"
    return f"{lo:g}–{hi:g}"


def compute_experience_bins(
    years: list[float], bins: list[float]
) -> tuple[list[tuple[str, int]], float | None, int]:
    """Bucket required-experience years into labelled histogram bins.

    ``bins`` are the config edges (``[0, 1, 3, 5, 10]``); an ``inf`` edge is
    appended so years ≥ the last edge are never silently dropped. Returns
    ``(labelled_counts, median, n_not_specified)`` where ``median`` is over
    non-NaN years (``None`` when there are none) and ``n_not_specified`` counts
    ``NaN`` entries.
    """
    arr = np.asarray(years, dtype=float)
    finite = arr[np.isfinite(arr)]
    n_not_specified = int(arr.size - finite.size)
    median = float(np.median(finite)) if finite.size else None
    edges = list(bins) + [np.inf]
    counts, _ = np.histogram(finite, bins=edges)
    labelled = [
        (_bin_label(edges[i], edges[i + 1]), int(counts[i]))
        for i in range(len(counts))
    ]
    return labelled, median, n_not_specified


def _spoke_label_lines(label: str) -> list[str]:
    return _wrap(label, _LABEL_WRAP, max_lines=2)


def _annotate_spoke(
    ax: PolarAxes,
    angle_rad: float,
    label: str,
    skills: list[str],
    *,
    label_fontsize: float = 10.0,
    skill_fontsize: float = 7.0,
) -> None:
    """Place an axis label and its top skills centered on a radar spoke."""
    label_lines = _spoke_label_lines(label)
    ax.text(
        angle_rad,
        _LABEL_R,
        "\n".join(label_lines),
        ha="center",
        va="center",
        fontsize=label_fontsize,
        fontweight="bold",
        color=_TEXT,
        linespacing=1.15,
    )

    skills_text = " · ".join(_skill_label(s) for s in skills[:3])
    skill_lines = _wrap(skills_text, _SKILL_WRAP, max_lines=2)
    if skill_lines:
        skill_r = _LABEL_R + _SKILL_GAP + 0.24 * (len(label_lines) - 1)
        ax.text(
            angle_rad,
            skill_r,
            "\n".join(skill_lines),
            ha="center",
            va="center",
            fontsize=skill_fontsize,
            style="italic",
            color=_MUTED,
            linespacing=1.1,
        )


def _draw_spider(
    ax: PolarAxes,
    model: dict[str, Any],
    characteristic_order: list[str],
    *,
    value_fontsize: float = 9.5,
    label_fontsize: float = 10.0,
    skill_fontsize: float = 7.0,
) -> None:
    """Draw one spider chart (polygon, markers, values, labels) onto an axes."""
    chars = _ordered_characteristics(model, characteristic_order)
    n = len(chars)
    if n == 0:
        ax.axis("off")
        return

    labels = [c["label_ru"] for c in chars]
    values = [c["proficiency"] for c in chars]
    skills = [c["top_skills"] for c in chars]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values_closed = values + [values[0]]
    angles_closed = angles.tolist() + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=_MUTED)
    ax.set_ylim(0, _PROFICIENCY_MAX)
    ax.set_xticks([])
    ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.9)
    ax.spines["polar"].set_color(_MUTED)
    ax.spines["polar"].set_linewidth(0.8)

    ax.plot(
        angles_closed, values_closed, linewidth=2.4, color=_PRIMARY,
        solid_capstyle="round", zorder=3,
    )
    ax.fill(angles_closed, values_closed, color=_PRIMARY, alpha=0.22, zorder=1)
    ax.plot(
        angles, values, "o", markersize=7, markerfacecolor="white",
        markeredgecolor=_PRIMARY, markeredgewidth=2.2, zorder=4,
    )

    # Value labels just outside each vertex (radial nudge in data units).
    for angle, value in zip(angles, values, strict=True):
        ax.text(
            angle,
            value + 0.3,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=value_fontsize,
            fontweight="bold",
            color=_PRIMARY,
            zorder=5,
        )

    for angle, label, top_skills in zip(angles, labels, skills, strict=True):
        _annotate_spoke(
            ax, angle, label, top_skills,
            label_fontsize=label_fontsize, skill_fontsize=skill_fontsize,
        )


def _apply_rc() -> None:
    rc = matplotlib.rcParams
    rc["font.family"] = "DejaVu Sans"
    rc["axes.edgecolor"] = _MUTED
    rc["axes.labelcolor"] = _TEXT
    rc["text.color"] = _TEXT
    rc["xtick.color"] = _MUTED
    rc["ytick.color"] = _MUTED
    rc["axes.titleweight"] = "bold"
    rc["figure.facecolor"] = "white"


def _figure_title(model: dict[str, Any], *, max_titles: int = 4) -> str:
    subtitle = f"{model['member_count']} вакансий"
    top_titles = model.get("top_job_titles") or []
    if top_titles:
        subtitle += " · " + ", ".join(top_titles[:max_titles])
    return subtitle


# ---------------------------------------------------------------------------
# Spider (radar) chart
# ---------------------------------------------------------------------------


def render_role_spider(
    model: dict[str, Any],
    characteristic_order: list[str],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Render a single 7-axis spider chart to a PNG."""
    _apply_rc()
    if not _ordered_characteristics(model, characteristic_order):
        raise ValueError("model has no characteristics to render")
    fig = plt.figure(figsize=(7.6, 7.6), dpi=dpi)
    ax = cast(PolarAxes, fig.add_subplot(111, projection="polar"))
    _draw_spider(ax, model, characteristic_order)

    fig.suptitle(model["role_label"], fontsize=16, fontweight="bold", color=_TEXT, y=0.97)
    fig.text(0.5, 0.925, _figure_title(model), ha="center", va="top", fontsize=9, color=_MUTED)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Competency model card: spider chart + achievement matrix
# ---------------------------------------------------------------------------


def render_role_card(
    model: dict[str, Any],
    characteristic_order: list[str],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Render the full competency model card (spider chart + level matrix)."""
    _apply_rc()
    chars = _ordered_characteristics(model, characteristic_order)
    if not chars:
        raise ValueError("model has no characteristics to render")
    char_ids = [c["characteristic_id"] for c in chars]

    fig = plt.figure(figsize=(13.2, 6.8), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.28)

    ax = cast(PolarAxes, fig.add_subplot(gs[0], projection="polar"))
    _draw_spider(ax, model, characteristic_order)

    levels = model.get("competency_levels") or []
    axh = fig.add_subplot(gs[1])
    if levels:
        _draw_achievement_matrix(axh, model, levels, char_ids)
        axh.set_title("Уровни компетенций", fontsize=11, fontweight="bold", pad=10)
    else:
        axh.axis("off")

    fig.suptitle(model["role_label"], fontsize=17, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.955, _figure_title(model, max_titles=5),
        ha="center", va="top", fontsize=9.5, color=_MUTED,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def _draw_achievement_matrix(
    ax: Axes,
    model: dict[str, Any],
    levels: list[dict[str, Any]],
    char_ids: list[str],
) -> None:
    """Draw the level × characteristic achievement matrix."""
    prof_by_id = {
        c["characteristic_id"]: c["proficiency"] for c in model["characteristics"]
    }

    # Rows bottom→top: lowest career level first (a ladder reading upward).
    rows = list(reversed(levels))
    n_rows = len(rows)
    n_cols = len(char_ids)

    grid = np.full((n_rows, n_cols), np.nan)
    for r, level in enumerate(rows):
        for c, cid in enumerate(char_ids):
            if cid in level["characteristics_achieved"]:
                grid[r, c] = prof_by_id.get(cid, 0.0)

    ax.imshow(
        np.ma.masked_invalid(grid),
        cmap=_ACHIEVEMENT_CMAP,
        vmin=_PROFICIENCY_MIN,
        vmax=_PROFICIENCY_MAX,
        aspect="auto",
        zorder=1,
    )

    for r in range(n_rows):
        for c in range(n_cols):
            if np.isnan(grid[r, c]):
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor=_NOT_ACHIEVED, edgecolor=_CELL_EDGE,
                        linewidth=0.7, zorder=0,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor="none", edgecolor="white", linewidth=0.9,
                        zorder=2,
                    )
                )

    for r, level in enumerate(rows):
        for c, cid in enumerate(char_ids):
            if cid in level["characteristics_achieved"]:
                ax.text(
                    c, r, f"{prof_by_id.get(cid, 0.0):.1f}",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white", zorder=3,
                )
            else:
                ax.text(
                    c, r, "·", ha="center", va="center",
                    fontsize=10, color="#B0B7BE", zorder=3,
                )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [_SHORT_LABELS.get(cid, cid) for cid in char_ids],
        fontsize=8.5, rotation=30, ha="right",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{lvl['label']}\n≥ {lvl['threshold']:.1f}" for lvl in rows],
        fontsize=8.3,
    )
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# Portfolio grid
# ---------------------------------------------------------------------------


def render_portfolio(
    models: list[dict[str, Any]],
    characteristic_order: list[str],
    output_path: Path | str,
    *,
    cols: int = 2,
    dpi: int = 300,
) -> Path:
    """Render a grid of spider charts for side-by-side role comparison."""
    _apply_rc()
    if not models:
        raise ValueError("no models to render")

    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.6 * cols, 6.9 * rows),
        subplot_kw={"projection": "polar"},
        dpi=dpi,
    )
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, model in enumerate(models):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        _draw_spider(
            ax, model, characteristic_order,
            value_fontsize=8.5, label_fontsize=9.0, skill_fontsize=6.5,
        )
        ax.set_title(
            f"{model['role_label']}\n{_figure_title(model, max_titles=3)}",
            fontsize=10.5, fontweight="bold", pad=22, color=_TEXT,
        )

    for idx in range(len(models), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Career trajectory (junior → C-level)
# ---------------------------------------------------------------------------

_AXIS_COLORS_7 = ["#2C5F8A", "#E07A5F", "#81B29A", "#F2CC8F", "#5A7D9A", "#B084CC", "#3AA0A8"]
_AXIS_COLORS = _AXIS_COLORS_7
_ROLE_COLORS = ["#2C5F8A", "#E07A5F", "#81B29A", "#F2CC8F", "#5A7D9A", "#B084CC"]


def render_career_trajectory(
    levels: list[CareerLevel],
    axis_ids: list[str],
    axis_labels: dict[str, str],
    output_path: Path | str,
    *,
    title: str = "Карьерная траектория: Junior → C-level",
    dpi: int = 300,
) -> Path:
    """Render the axis shift across career levels as a multi-line chart.

    Each line is one competency axis; x-axis is career seniority (junior →
    C-level), y-axis is proficiency 1–5. Every axis grows monotonically; the
    role's dominant pair stays ahead of the rest at every level and never
    declines at C-level.
    """
    _apply_rc()
    x = list(range(len(levels)))
    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=dpi)

    for i, cid in enumerate(axis_ids):
        y = [level["axes"].get(cid, np.nan) for level in levels]
        ax.plot(
            x, y, marker="o", markersize=4.5, linewidth=2.2,
            color=_AXIS_COLORS[i % len(_AXIS_COLORS)],
            label=axis_labels.get(cid, cid),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [level["label"] for level in levels],
        fontsize=8.2, rotation=25, ha="right",
    )
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Уровень владения", fontsize=9)
    ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def render_career_heatmap(
    levels: list[CareerLevel],
    axis_ids: list[str],
    axis_labels: dict[str, str],
    output_path: Path | str,
    *,
    title: str = "Профиль компетенций по уровням",
    dpi: int = 300,
) -> Path:
    """Render the level × axis proficiency heatmap (junior → C-level)."""
    _apply_rc()
    n_rows = len(levels)
    n_cols = len(axis_ids)

    grid = np.array(
        [[level["axes"].get(cid, np.nan) for cid in axis_ids] for level in levels],
        dtype=float,
    )
    # Rows bottom→top: junior first, C-level last (a ladder reading upward).
    grid = np.flipud(grid)
    row_labels = [level["label"] for level in reversed(levels)]

    fig, ax = plt.subplots(figsize=(9.6, 6.6), dpi=dpi)
    ax.imshow(
        grid, cmap=_ACHIEVEMENT_CMAP, vmin=_PROFICIENCY_MIN, vmax=_PROFICIENCY_MAX,
        aspect="auto",
    )

    for r in range(n_rows):
        for c in range(n_cols):
            v = grid[r, c]
            ax.text(
                c, r, f"{v:.1f}", ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color="white" if v >= 3.5 else "#1F2937",
            )
            ax.add_patch(
                Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none",
                          edgecolor="white", linewidth=0.9)
            )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [axis_labels.get(cid, cid) for cid in axis_ids],
        fontsize=8.3, rotation=28, ha="right",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8.6)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def render_comparison(
    models: list[dict[str, Any]],
    characteristic_order: list[str],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Overlay all roles' spider profiles onto one axes with a legend."""
    _apply_rc()
    chars = _ordered_characteristics(models[0], characteristic_order)
    n = len(chars)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    labels = [c["label_ru"] for c in chars]

    fig = plt.figure(figsize=(8.0, 7.6), dpi=dpi)
    ax = cast(PolarAxes, fig.add_subplot(111, projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=_MUTED)
    ax.set_ylim(0, _PROFICIENCY_MAX)
    ax.set_xticks([])
    ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.9)
    ax.spines["polar"].set_color(_MUTED)
    ax.spines["polar"].set_linewidth(0.8)

    for i, model in enumerate(models):
        values = [c["proficiency"] for c in _ordered_characteristics(model, characteristic_order)]
        color = _ROLE_COLORS[i % len(_ROLE_COLORS)]
        ax.plot(
            angles.tolist() + [angles[0]], values + [values[0]],
            linewidth=2.0, color=color, label=model["role_label"], zorder=3,
        )
        ax.fill(
            angles.tolist() + [angles[0]], values + [values[0]],
            color=color, alpha=0.06, zorder=1,
        )

    for angle, label in zip(angles, labels, strict=True):
        ax.text(
            angle, _LABEL_R, "\n".join(_spoke_label_lines(label)),
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=_TEXT,
        )
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def render_summary(
    role_labels: list[str],
    member_counts: list[int],
    skill_counts: list[int],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Render the summary bar charts (members and skills per role)."""
    _apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=dpi)
    y = np.arange(len(role_labels))[::-1]

    ax = axes[0]
    ax.barh(y, member_counts, color=_PRIMARY, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(role_labels, fontsize=8.8)
    ax.set_xlabel("Вакансий", fontsize=9)
    ax.set_title("Число вакансий по ролям", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", color=_GRID, linewidth=0.7, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.barh(y, skill_counts, color="#E07A5F", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(role_labels, fontsize=8.8)
    ax.set_xlabel("Навыков", fontsize=9)
    ax.set_title("Навыков по ролям", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", color=_GRID, linewidth=0.7, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Role transition map & leadership gap
# ---------------------------------------------------------------------------


def render_transition_map(
    roles: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    axis_labels: dict[str, str],
    output_path: Path | str,
    *,
    leadership_label: str = "Руководитель лаборатории",
    leadership_edges: list[dict[str, Any]] | None = None,
    dpi: int = 300,
) -> Path:
    """Render the role→role transition map as a network graph.

    Roles sit on a circle; leadership (a pseudo-node, not a base role) sits at
    the centre. An edge connects two roles when they share exactly one dominant
    competence (a "pivot"); the edge is labelled with that shared competence.
    Dashed amber edges point from every role toward leadership.
    """
    _apply_rc()
    n = len(roles)
    angles = np.linspace(90.0, 90.0 + 360.0, n, endpoint=False)
    pos: dict[int, tuple[float, float]] = {}
    for i, deg in enumerate(angles):
        a = np.deg2rad(deg)
        pos[i] = (np.cos(a), np.sin(a))
    center = (0.0, 0.0)

    fig, ax = plt.subplots(figsize=(11.5, 9.4), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    # Undirected role→role edges, labelled with the shared competence.
    drawn: set[tuple[int, int]] = set()
    for e in transitions:
        key = tuple(sorted((e["source"], e["target"])))
        if key in drawn:
            continue
        drawn.add(key)
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        ax.plot([x0, x1], [y0, y1], color="#9AA5B1", linewidth=1.4, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        shared = axis_labels.get(e["shared_axis"], e["shared_axis"] or "")
        ax.text(
            mx, my, shared, ha="center", va="center", fontsize=7.2,
            color=_MUTED, zorder=2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.6},
        )

    # Leadership edges (every role → centre).
    for e in (leadership_edges or []):
        x0, y0 = pos[e["source"]]
        ax.plot([x0, center[0]], [y0, center[1]], color=_ACCENT,
                linewidth=1.3, linestyle="--", alpha=0.75, zorder=1)

    # Role nodes + outward labels.
    for i in range(n):
        x, y = pos[i]
        ax.scatter([x], [y], s=1500, color=_PRIMARY, zorder=3)
        deg = angles[i]
        a = np.deg2rad(deg)
        cos, sin = np.cos(a), np.sin(a)
        lx, ly = 1.42 * cos, 1.42 * sin
        ha = "center" if abs(cos) < 0.32 else ("left" if cos > 0 else "right")
        va = "center" if abs(sin) < 0.32 else ("bottom" if sin > 0 else "top")
        ax.text(lx, ly, roles[i]["label"], ha=ha, va=va, fontsize=9.8,
                fontweight="bold", color=_TEXT, zorder=4)

    # Leadership centre node.
    ax.scatter([center[0]], [center[1]], s=2400, color=_ACCENT, zorder=3)
    ax.text(center[0], center[1] - 0.24, leadership_label, ha="center",
            va="top", fontsize=10.5, fontweight="bold", color="#8A5A00", zorder=4)

    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-1.95, 1.95)
    ax.set_title("Карта перехода между ролями", fontsize=13,
                 fontweight="bold", pad=14)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def render_leadership_gap(
    roles: list[dict[str, Any]],
    leadership_edges: list[dict[str, Any]],
    axis_labels: dict[str, str],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Render the gap to leadership (management + domain) per role."""
    _apply_rc()
    labels = [r["label"] for r in roles]
    mgmt = [e["gap"].get("management", 0.0) for e in leadership_edges]
    dom = [e["gap"].get("domain_knowledge", 0.0) for e in leadership_edges]
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=dpi)
    ax.barh(y + 0.2, mgmt, height=0.36, color=_PRIMARY,
            label=axis_labels.get("management", "management"))
    ax.barh(y - 0.2, dom, height=0.36, color=_ACCENT,
            label=axis_labels.get("domain_knowledge", "domain_knowledge"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlabel("Разрыв (Δ) до уровня руководителя", fontsize=9)
    ax.set_title("Путь в руководство: что нужно дорастить",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Experience histogram
# ---------------------------------------------------------------------------


def render_role_experience(
    role_label: str,
    years: list[float],
    output_path: Path | str,
    *,
    bins: list[float],
    dpi: int = 300,
) -> Path:
    """Render the required-experience histogram (bins + "не указан" + median)."""
    _apply_rc()
    labelled, median, n_not_specified = compute_experience_bins(years, bins)

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=dpi)
    if not years:
        ax.text(
            0.5, 0.5, "нет данных", ha="center", va="center",
            fontsize=13, color=_MUTED, transform=ax.transAxes,
        )
        ax.axis("off")
    else:
        bar_labels = [r for r, _ in labelled] + ["не указан"]
        bar_counts = [c for _, c in labelled] + [n_not_specified]
        colors = [_PRIMARY] * len(labelled) + [_MUTED]
        y = np.arange(len(bar_labels))[::-1]
        ax.barh(y, bar_counts, color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(bar_labels, fontsize=8.8)
        ax.set_xlabel("Вакансий", fontsize=9)
        ax.set_title(f"Требуемый опыт: {role_label}", fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, axis="x", color=_GRID, linewidth=0.7, alpha=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if median is not None:
            ax.annotate(
                f"медиана: {median:.1f} лет",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color=_MUTED,
            )

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Skills bars (top-N, segmented by axis membership)
# ---------------------------------------------------------------------------


def render_role_skills(
    role_label: str,
    skills: list[dict[str, Any]],
    axis_labels: dict[str, str],
    output_path: Path | str,
    *,
    top_n: int = 15,
    dpi: int = 300,
) -> Path:
    """Render the top-N skills as horizontal bars segmented by axis weight.

    Bars preserve the input order (phase 6 pre-sorts by ``tfidf_weight`` desc);
    each bar is split left-to-right into axis segments whose length is the raw
    NLI score for that axis.
    """
    _apply_rc()
    skills = list(skills)[:top_n]
    axis_order = list(axis_labels)
    color_by_axis = {
        cid: _AXIS_COLORS_7[i % len(_AXIS_COLORS_7)]
        for i, cid in enumerate(axis_order)
    }

    fig, ax = plt.subplots(
        figsize=(10.0, max(3.4, 0.34 * len(skills) + 1.2)), dpi=dpi
    )
    if not skills:
        ax.text(
            0.5, 0.5, "нет данных", ha="center", va="center",
            fontsize=13, color=_MUTED, transform=ax.transAxes,
        )
        ax.axis("off")
    else:
        y = np.arange(len(skills))[::-1]
        for i, sk in enumerate(skills):
            left = 0.0
            for cid, score in (sk.get("axis_weights") or {}).items():
                ax.barh(
                    y[i], float(score), left=left, height=0.6,
                    color=color_by_axis.get(cid, _MUTED),
                )
                left += float(score)
        ax.set_yticks(y)
        ax.set_yticklabels([_skill_label(sk) for sk in skills], fontsize=8.4)
        ax.set_xlabel("Вклад оси (NLI)", fontsize=9)
        ax.set_title(f"Ключевые навыки: {role_label}", fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, axis="x", color=_GRID, linewidth=0.7, alpha=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        present = {cid for sk in skills for cid in (sk.get("axis_weights") or {})}
        handles = [
            Patch(facecolor=color_by_axis[cid], label=axis_labels.get(cid, cid))
            for cid in axis_order if cid in present
        ]
        if handles:
            ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Soft-competence spider (reuses the generic radar renderer)
# ---------------------------------------------------------------------------


def render_role_soft_spider(
    model: dict[str, Any],
    soft_order: list[str],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Render a soft-competence spider by reusing the generic spider.

    The soft competences are read from ``model["soft_competences"]`` and reshaped
    into the same ``characteristics`` form the hard-competence spider consumes.
    """
    soft = model.get("soft_competences") or []
    by_id = {c["soft_id"]: c for c in soft}
    characteristics = [
        {
            "characteristic_id": sid,
            "label_ru": by_id[sid]["label_ru"],
            "proficiency": by_id[sid]["proficiency"],
            "top_skills": [],
        }
        for sid in soft_order
        if sid in by_id
    ]
    soft_model: dict[str, Any] = {
        "role_label": model["role_label"],
        "member_count": model["member_count"],
        "top_job_titles": model.get("top_job_titles"),
        "characteristics": characteristics,
    }
    return render_role_spider(soft_model, soft_order, output_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Plotly (interactive HTML) renderers — used by the HTML report
# ---------------------------------------------------------------------------
# The report embeds interactive charts as SVG/JS divs instead of static PNGs.
# Each renderer returns a self-contained ``<div>`` via
# ``fig.to_html(include_plotlyjs=False, full_html=False, div_id=...)``; the
# caller injects Plotly.js exactly once — the first chart in the document
# carries ``include_plotlyjs="cdn"``. The polar charts match the animated
# spider's layout: axes clockwise from the top (``rotation=90``,
# ``direction="clockwise"``) with radial range 1–5.


def _hex_alpha(color: str, alpha: float) -> str:
    """Convert a ``#RRGGBB`` color to an ``rgba(r,g,b,a)`` string (plotly-safe)."""
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _polar_layout(
    labels: list[str],
    *,
    height: int,
    title: str | None = None,
) -> dict[str, Any]:
    """Shared Plotly polar layout: 7 (or 4) labelled axes, radial range 1–5."""
    theta_deg = [round(i * 360 / len(labels), 2) for i in range(len(labels))]
    layout: dict[str, Any] = {
        "polar": {
            "radialaxis": {
                "visible": True, "range": [1, 5], "tickvals": [1, 2, 3, 4, 5],
                "tickfont": {"size": 9, "color": _MUTED},
            },
            "angularaxis": {
                "direction": "clockwise", "rotation": 90,
                "tickmode": "array", "tickvals": theta_deg, "ticktext": labels,
                "tickfont": {"size": 11, "color": _TEXT},
            },
        },
        "showlegend": False,
        "height": height,
        "margin": {"l": 80, "r": 80, "t": 50 if title else 30, "b": 50},
    }
    if title:
        layout["title"] = {"text": title, "font": {"size": 14, "color": _TEXT}}
    return layout


def _empty_figure_html(
    include_plotlyjs: bool | str,
    div_id: str | None,
) -> str:
    """A minimal "нет данных" Plotly div for roles without data."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text="нет данных", showarrow=False,
        xref="paper", yref="paper", x=0.5, y=0.5,
        font={"size": 13, "color": _MUTED},
    )
    fig.update_layout(
        height=180, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False}, yaxis={"visible": False},
    )
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=div_id)


def render_role_spider_html(
    model: dict[str, Any],
    characteristic_order: list[str],
    *,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
) -> str:
    """Render a single spider (radar) chart as an interactive Plotly div.

    Layout matches the animated spider in the report: axes clockwise from the
    top, radial range 1–5.
    """
    import plotly.graph_objects as go

    chars = _ordered_characteristics(model, characteristic_order)
    if not chars:
        raise ValueError("model has no characteristics to render")
    labels = [c["label_ru"] for c in chars]
    values = [float(c["proficiency"]) for c in chars]
    n = len(labels)
    theta_deg = [round(i * 360 / n, 2) for i in range(n)]
    hover = [f"{label}: {value:.2f}" for label, value in zip(labels, values, strict=True)]

    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=theta_deg + [360],
            fill="toself",
            mode="lines+markers",
            line={"color": _PRIMARY, "width": 2.4},
            marker={"size": 7, "color": "white",
                        "line": {"color": _PRIMARY, "width": 2.2}},
            fillcolor="rgba(44,95,138,0.22)",
            customdata=hover + [hover[0]],
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(**_polar_layout(labels, height=420))
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=div_id)


def render_role_soft_spider_html(
    model: dict[str, Any],
    soft_order: list[str],
    *,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
) -> str:
    """Render a soft-competence spider as an interactive Plotly div.

    Reuses :func:`render_role_spider_html` after reshaping
    ``model["soft_competences"]`` into the same ``characteristics`` form the
    hard-competence spider consumes.
    """
    soft = model.get("soft_competences") or []
    by_id = {c["soft_id"]: c for c in soft}
    characteristics = [
        {
            "characteristic_id": sid,
            "label_ru": by_id[sid]["label_ru"],
            "proficiency": by_id[sid]["proficiency"],
            "top_skills": [],
        }
        for sid in soft_order
        if sid in by_id
    ]
    soft_model: dict[str, Any] = {
        "role_label": model["role_label"],
        "member_count": model["member_count"],
        "top_job_titles": model.get("top_job_titles"),
        "characteristics": characteristics,
    }
    return render_role_spider_html(
        soft_model, soft_order, include_plotlyjs=include_plotlyjs,
        div_id=div_id,
    )


def render_role_experience_html(
    role_label: str,
    labelled: list[tuple[str, int]],
    median: float | None,
    n_not_specified: int,
    *,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
) -> str:
    """Render the required-experience histogram as an interactive Plotly div.

    ``labelled`` is the ``(range_label, count)`` bucket list produced by
    :func:`compute_experience_bins` (or the model's precomputed bins).
    """
    import plotly.graph_objects as go

    bar_labels = [r for r, _ in labelled]
    bar_counts = [c for _, c in labelled]
    colors = [_PRIMARY] * len(labelled)
    if n_not_specified:
        bar_labels.append("не указан")
        bar_counts.append(n_not_specified)
        colors.append(_MUTED)

    if not bar_counts:
        return _empty_figure_html(include_plotlyjs, div_id)

    fig = go.Figure(
        go.Bar(
            x=bar_counts, y=bar_labels, orientation="h",
            marker={"color": colors},
            hovertemplate="%{x} вакансий<extra>%{y}</extra>",
        )
    )
    if median is not None:
        fig.add_annotation(
            text=f"медиана: {median:.1f} лет", showarrow=False,
            xref="paper", yref="paper", x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            font={"size": 10, "color": _MUTED},
        )
    fig.update_layout(
        title={"text": f"Требуемый опыт: {role_label}",
                   "font": {"size": 13, "color": _TEXT}},
        xaxis={"title": "Вакансий", "gridcolor": _GRID},
        yaxis={"autorange": "reversed"},
        showlegend=False,
         height=max(420, 26 * len(bar_labels) + 140),
         margin={"l": 120, "r": 30, "t": 50, "b": 40},
    )
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=div_id)


def render_role_skills_html(
    role_label: str,
    skills: list[dict[str, Any]],
    axis_labels: dict[str, str],
    *,
    top_n: int = 15,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
) -> str:
    """Render the top-N skills as horizontal stacked bars (axis segments).

    Each bar is one skill; segments are the per-axis NLI weights, coloured by
    axis — the interactive equivalent of the static matplotlib version.
    """
    import plotly.graph_objects as go

    skills = list(skills)[:top_n]
    axis_order = list(axis_labels)
    color_by_axis = {
        cid: _AXIS_COLORS_7[i % len(_AXIS_COLORS_7)]
        for i, cid in enumerate(axis_order)
    }

    if not skills:
        return _empty_figure_html(include_plotlyjs, div_id)

    names = [_skill_label(sk) for sk in skills]
    fig = go.Figure()
    for cid in axis_order:
        xs = [float((sk.get("axis_weights") or {}).get(cid, 0.0)) for sk in skills]
        if not any(x > 0.0 for x in xs):
            continue
        fig.add_trace(go.Bar(
            x=xs, y=names, orientation="h",
            name=axis_labels.get(cid, cid),
            marker={"color": color_by_axis.get(cid, _MUTED)},
            hovertemplate="%{y}: %{x:.2f}<extra>"
            + axis_labels.get(cid, cid) + "</extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title={"text": f"Ключевые навыки: {role_label}",
                   "font": {"size": 13, "color": _TEXT}},
        xaxis={"title": "Вклад оси (NLI)", "gridcolor": _GRID},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                    "xanchor": "right", "x": 1, "font": {"size": 10}},
         height=max(420, 22 * len(names) + 160),
         margin={"l": 180, "r": 30, "t": 60, "b": 40},
    )
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=div_id)


def render_comparison_html(
    models: list[dict[str, Any]],
    characteristic_order: list[str],
    *,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
) -> str:
    """Overlay all roles' spider profiles onto one interactive Plotly chart."""
    import plotly.graph_objects as go

    if not models:
        raise ValueError("no models to compare")
    chars = _ordered_characteristics(models[0], characteristic_order)
    labels = [c["label_ru"] for c in chars]
    n = len(labels)
    theta_deg = [round(i * 360 / n, 2) for i in range(n)]

    fig = go.Figure()
    for i, model in enumerate(models):
        values = [
            float(c["proficiency"])
            for c in _ordered_characteristics(model, characteristic_order)
        ]
        color = _ROLE_COLORS[i % len(_ROLE_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=theta_deg + [360],
            fill="toself",
            mode="lines",
            line={"color": color, "width": 2.0},
            fillcolor=_hex_alpha(color, 0.08),
            name=model["role_label"],
            hovertemplate="%{theta}: %{r:.2f}<extra>"
            + model["role_label"] + "</extra>",
        ))
    layout = _polar_layout(labels, height=480)
    layout["showlegend"] = True
    layout["legend"] = {"x": 1.02, "y": 1.0, "font": {"size": 11}}
    layout["margin"] = {"l": 80, "r": 180, "t": 30, "b": 50}
    fig.update_layout(**layout)
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=div_id)


def render_summary_html(
    role_labels: list[str],
    member_counts: list[int],
    skill_counts: list[int],
    *,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
) -> str:
    """Render the summary bar charts (vacancies and skills per role) as Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Число вакансий по ролям", "Навыков по ролям"),
        horizontal_spacing=0.28,
    )
    fig.add_trace(go.Bar(
        x=member_counts, y=role_labels, orientation="h",
        marker={"color": _PRIMARY},
        hovertemplate="%{x} вакансий<extra>%{y}</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=skill_counts, y=role_labels, orientation="h",
        marker={"color": "#E07A5F"},
        hovertemplate="%{x} навыков<extra>%{y}</extra>",
    ), row=1, col=2)
    fig.update_layout(
        showlegend=False,
        height=max(300, 24 * len(role_labels) + 160),
        margin={"l": 200, "r": 30, "t": 50, "b": 30},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(gridcolor=_GRID)
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=div_id)


def render_leadership_gap_html(
    roles: list[dict[str, Any]],
    leadership_edges: list[dict[str, Any]],
    axis_labels: dict[str, str],
    *,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
) -> str:
    """Render the gap to leadership (management + domain) per role as Plotly."""
    import plotly.graph_objects as go

    labels = [r["label"] for r in roles]
    mgmt = [e["gap"].get("management", 0.0) for e in leadership_edges]
    dom = [e["gap"].get("domain_knowledge", 0.0) for e in leadership_edges]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=mgmt, orientation="h",
        name=axis_labels.get("management", "management"),
        marker={"color": _PRIMARY},
        hovertemplate="%{x:.2f}<extra>"
        + axis_labels.get("management", "management") + "</extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=dom, orientation="h",
        name=axis_labels.get("domain_knowledge", "domain_knowledge"),
        marker={"color": "#F4A261"},
        hovertemplate="%{x:.2f}<extra>"
        + axis_labels.get("domain_knowledge", "domain_knowledge") + "</extra>",
    ))
    fig.update_layout(
        barmode="group",
        title={"text": "Путь в руководство: что нужно дорастить",
                   "font": {"size": 14, "color": _TEXT}},
        xaxis={"title": "Разрыв (Δ) до уровня руководителя", "gridcolor": _GRID},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                    "xanchor": "right", "x": 1},
        height=max(300, 26 * len(labels) + 160),
        margin={"l": 200, "r": 30, "t": 60, "b": 40},
    )
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False,
                       div_id=div_id)
