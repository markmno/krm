"""Spider/radar chart comparing skill profiles across specialties.

Uses Plotly go.Scatterpolar to render multi-axis radar charts with
differentiating skills on axes and one trace per specialty.
"""

from __future__ import annotations

import plotly.graph_objects as go

from hh_competency.analysis.compare import SkillComparator
from hh_competency.storage.db import Database

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


def generate_radar_chart(
    db: Database,
    specialties: list[str],
    top_n_skills: int = 8,
    output_path: str | None = None,
) -> go.Figure:
    """Generate a spider/radar chart comparing skill profiles across specialties.

    Axes are the top N most differentiating skills across all specialty pairs,
    extracted via SkillComparator.differentiating_skills. Lines show the frequency
    of each differentiating skill in each specialty.

    Args:
        db: Database instance with skill frequency data.
        specialties: List of specialty keys to compare (e.g. ['python_dev', 'data_science']).
        top_n_skills: Number of differentiating skill axes on the radar.
        output_path: If provided, save self-contained HTML to this path.

    Returns:
        Plotly Figure (go.Figure radar chart).
    """
    if len(specialties) < 2:
        return _empty_figure("Нужно минимум 2 специальности для радарной диаграммы")

    # -- Collect differentiating skills across all specialty pairs --
    comparator = SkillComparator(db)
    all_diff: dict[str, float] = {}

    for i in range(len(specialties)):
        for j in range(i + 1, len(specialties)):
            diff = comparator.differentiating_skills(
                specialties[i], specialties[j], top_n=20
            )
            for lemma, score in diff["a_vs_b"]:
                all_diff[lemma] = max(all_diff.get(lemma, 0.0), score)
            for lemma, score in diff["b_vs_a"]:
                all_diff[lemma] = max(all_diff.get(lemma, 0.0), score)

    if not all_diff:
        return _empty_figure("Нет данных по навыкам ни для одной специальности")

    # Pick top N by differentiation score
    sorted_diff = sorted(all_diff.items(), key=lambda x: x[1], reverse=True)
    diff_lemmas = [lemma for lemma, _score in sorted_diff[:top_n_skills]]

    # -- Get frequencies for each specialty --
    all_freqs = db.get_skill_frequencies_multi(specialties)
    spec_freq_map: dict[str, dict[str, float]] = {s: {} for s in specialties}
    for sf in all_freqs:
        if sf.lemma in diff_lemmas and sf.specialty in spec_freq_map:
            spec_freq_map[sf.specialty][sf.lemma] = float(sf.frequency)

    # -- Build radar traces --
    fig = go.Figure()

    for idx, spec in enumerate(specialties):
        freq_map = spec_freq_map[spec]
        values = [freq_map.get(lemma, 0.0) for lemma in diff_lemmas]
        color = _COLORS[idx % len(_COLORS)]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=diff_lemmas,
                name=spec,
                fill="toself",
                opacity=0.3,
                line={"color": color, "width": 2},
                marker={"color": color, "size": 6},
                hovertemplate="<b>%{theta}</b><br>%{fullData.name}: %{r}<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": "Сравнение компетенций по специальностям",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        polar={
            "radialaxis": {
                "visible": True,
                "showticklabels": True,
                "gridcolor": "#dddddd",
            },
            "angularaxis": {"gridcolor": "#dddddd"},
        },
        width=1000,
        height=600,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
        margin={"t": 60, "b": 80, "l": 80, "r": 80},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def radar_data_json(
    db: Database,
    specialties: list[str],
    top_n_skills: int = 8,
) -> dict:
    """Export radar chart data as a JSON-serializable dict for external consumption.

    Args:
        db: Database instance.
        specialties: List of specialty keys.
        top_n_skills: Number of differentiating skill axes.

    Returns:
        Dict with keys: 'skills' (list of skill lemmas), 'specialties' (dict
        mapping specialty → list of frequency values), 'scores' (list of
        differentiation scores).
    """
    if len(specialties) < 2:
        return {"skills": [], "specialties": {}, "scores": []}

    comparator = SkillComparator(db)
    all_diff: dict[str, float] = {}

    for i in range(len(specialties)):
        for j in range(i + 1, len(specialties)):
            diff = comparator.differentiating_skills(
                specialties[i], specialties[j], top_n=20
            )
            for lemma, score in diff["a_vs_b"]:
                all_diff[lemma] = max(all_diff.get(lemma, 0.0), score)
            for lemma, score in diff["b_vs_a"]:
                all_diff[lemma] = max(all_diff.get(lemma, 0.0), score)

    if not all_diff:
        return {"skills": [], "specialties": {}, "scores": []}

    sorted_diff = sorted(all_diff.items(), key=lambda x: x[1], reverse=True)
    diff_lemmas = [lemma for lemma, _score in sorted_diff[:top_n_skills]]
    scores = [score for _lemma, score in sorted_diff[:top_n_skills]]

    all_freqs = db.get_skill_frequencies_multi(specialties)
    spec_freq_map: dict[str, dict[str, float]] = {s: {} for s in specialties}
    for sf in all_freqs:
        if sf.lemma in diff_lemmas and sf.specialty in spec_freq_map:
            spec_freq_map[sf.specialty][sf.lemma] = float(sf.frequency)

    specialties_data: dict[str, list[float]] = {}
    for spec in specialties:
        freq_map = spec_freq_map[spec]
        specialties_data[spec] = [freq_map.get(lemma, 0.0) for lemma in diff_lemmas]

    return {
        "skills": diff_lemmas,
        "specialties": specialties_data,
        "scores": scores,
    }
