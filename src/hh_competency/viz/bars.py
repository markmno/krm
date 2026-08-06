"""Bar charts for skill frequencies and role profiles.

Horizontal bar charts for top skills, grouped comparisons across specialties,
and role profile breakdowns with defining vs supporting skills.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hh_competency.storage.db import Database
from hh_competency.storage.models import SkillFrequency

# -- Consistent color palette (Plotly default 10) --
_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _empty_figure(message: str) -> go.Figure:
    """Return a figure with a centered annotation message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "#888888"},
    )
    fig.update_layout(
        width=1000,
        height=600,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def generate_skill_bar_chart(
    db: Database,
    specialty: str,
    top_n: int = 20,
    output_path: str | None = None,
) -> go.Figure:
    """Generate a horizontal bar chart of the top N skills for a specialty.

    Args:
        db: Database instance.
        specialty: Specialty key.
        top_n: Number of top skills to show.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with horizontal bars sorted descending by frequency.
    """
    skills: list[SkillFrequency] = db.get_skill_frequencies(specialty, top_n=top_n)

    if not skills:
        return _empty_figure(f"Нет данных по навыкам для специальности: {specialty}")

    # Sort descending (already sorted by DB, but ensure)
    skills.sort(key=lambda s: s.frequency, reverse=True)

    lemmas = [s.lemma for s in skills]
    freqs = [s.frequency for s in skills]
    counts = [s.vacancy_count for s in skills]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=freqs,
            y=lemmas,
            orientation="h",
            marker={"color": _COLORS[0], "line": {"color": _COLORS[0], "width": 1}},
            hovertemplate="<b>%{y}</b><br>Частота: %{x}<br>Вакансий: %{customdata}<extra></extra>",
            customdata=counts,
        )
    )

    fig.update_layout(
        title={
            "text": f"Топ-{top_n} навыков: {specialty}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="Частота",
        yaxis={"autorange": "reversed"},  # Highest frequency at top
        width=1000,
        height=max(600, top_n * 30),
        margin={"t": 60, "b": 60, "l": 150, "r": 40},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def generate_comparison_bar_chart(
    db: Database,
    specialties: list[str],
    top_n: int = 15,
    output_path: str | None = None,
) -> go.Figure:
    """Generate a grouped horizontal bar chart comparing top skills across specialties.

    Args:
        db: Database instance.
        specialties: List of specialty keys to compare.
        top_n: Number of top skills to show from each specialty.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with grouped horizontal bars, one group per specialty.
    """
    if not specialties:
        return _empty_figure("Не указаны специальности для сравнения")

    # Collect all unique skill lemmas across all specialties
    all_skills: dict[str, str] = {}  # lemma -> first-seen specialty
    spec_skills: dict[str, dict[str, int]] = {}

    for _idx, spec in enumerate(specialties):
        skills = db.get_skill_frequencies(spec, top_n=top_n)
        spec_skills[spec] = {}
        for sf in skills:
            spec_skills[spec][sf.lemma] = sf.frequency
            if sf.lemma not in all_skills:
                all_skills[sf.lemma] = spec
            # Use color of last specialty that has this skill
            all_skills[sf.lemma] = spec

    if not all_skills:
        return _empty_figure("Нет данных по навыкам ни для одной специальности")

    # To keep bars manageable, use the union of top skills sorted by max frequency
    max_freq: dict[str, int] = {}
    for lemma in all_skills:
        max_freq[lemma] = max(spec_skills.get(spec, {}).get(lemma, 0) for spec in specialties)

    sorted_lemmas = sorted(max_freq.items(), key=lambda x: x[1], reverse=True)
    # Take at most top_n * len(specialties) to avoid overcrowding
    display_lemmas = [lemma for lemma, _f in sorted_lemmas[: top_n * len(specialties)]]

    fig = go.Figure()

    for idx, spec in enumerate(specialties):
        s_map = spec_skills[spec]
        values = [s_map.get(lemma, 0) for lemma in display_lemmas]
        color = _COLORS[idx % len(_COLORS)]

        fig.add_trace(
            go.Bar(
                name=spec,
                x=values,
                y=display_lemmas,
                orientation="h",
                marker={"color": color},
                hovertemplate="<b>%{y}</b><br>%{data.name}: %{x}<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": "Сравнение навыков по специальностям",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="Частота",
        yaxis={"autorange": "reversed"},
        barmode="group",
        width=1000,
        height=max(600, len(display_lemmas) * 25),
        margin={"t": 60, "b": 60, "l": 150, "r": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def generate_role_profile_chart(
    db: Database,
    specialty: str,
    output_path: str | None = None,
) -> go.Figure:
    """Generate a bar chart showing discovered role profiles with defining vs supporting skills.

    Creates one subplot per role profile, displaying defining skills with
    darker color and supporting skills with lighter color.

    Args:
        db: Database instance.
        specialty: Specialty key.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with subplots for each role profile.
    """
    profiles = db.get_role_profiles(specialty)

    if not profiles:
        return _empty_figure(f"Не обнаружено ролевых профилей для: {specialty}")

    n_profiles = len(profiles)
    fig = make_subplots(
        rows=n_profiles,
        cols=1,
        subplot_titles=[p.role_name for p in profiles],
        vertical_spacing=0.08,
    )

    for idx, profile in enumerate(profiles):
        row = idx + 1

        # Defining skills (darker)
        def_lemmas = [lemma for lemma, _w in profile.defining_skills]
        def_weights = [w for _lemma, w in profile.defining_skills]
        color_idx = idx % len(_COLORS)

        fig.add_trace(
            go.Bar(
                name=f"{profile.role_name} — определяющие",
                x=def_weights,
                y=def_lemmas,
                orientation="h",
                marker={"color": _COLORS[color_idx]},
                hovertemplate="<b>%{y}</b><br>Вес: %{x:.1f}<extra></extra>",
                legendgroup=profile.role_name,
                showlegend=(idx == 0),
            ),
            row=row,
            col=1,
        )

        # Supporting skills (lighter — same hue, reduced opacity)
        sup_lemmas = [lemma for lemma, _w in profile.supporting_skills]
        sup_weights = [w for _lemma, w in profile.supporting_skills]

        if sup_lemmas:
            fig.add_trace(
                go.Bar(
                    name=f"{profile.role_name} — поддерживающие",
                    x=sup_weights,
                    y=sup_lemmas,
                    orientation="h",
                    marker={"color": _COLORS[color_idx], "opacity": 0.45},
                    hovertemplate="<b>%{y}</b><br>Вес: %{x:.1f}<extra></extra>",
                    legendgroup=profile.role_name,
                    showlegend=(idx == 0),
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        title={
            "text": f"Ролевые профили: {specialty}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        barmode="group",
        width=1000,
        height=max(600, n_profiles * 300),
        margin={"t": 60, "b": 60, "l": 150, "r": 40},
        showlegend=True,
    )

    # Reverse y-axis for each subplot so highest weight is at top
    for row_idx in range(1, n_profiles + 1):
        fig.update_yaxes(autorange="reversed", row=row_idx, col=1)

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig
