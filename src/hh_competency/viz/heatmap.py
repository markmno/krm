"""Heatmaps for skill co-occurrence within specialties and overlap between specialties.

Uses Plotly go.Heatmap to visualize skill co-occurrence matrices and
Jaccard overlap scores between specialty skill sets.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from hh_competency.analysis.compare import SkillComparator
from hh_competency.nlp.pipeline import NLPPipeline
from hh_competency.storage.db import Database
from hh_competency.storage.models import SkillFrequency


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


def generate_cooccurrence_heatmap(
    db: Database,
    specialty: str,
    top_n: int = 15,
    output_path: str | None = None,
) -> go.Figure:
    """Generate a heatmap of skill co-occurrence within a specialty.

    Rows and columns are the top N skills. Each cell [i, j] shows how many
    vacancies contain both skill i and skill j simultaneously.

    Builds the co-occurrence matrix by running NLPPipeline over all vacancy
    descriptions and counting pairwise co-occurrences.

    Args:
        db: Database instance.
        specialty: Specialty key.
        top_n: Number of top skills to include in the heatmap.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with a go.Heatmap trace.
    """
    descriptions = db.get_vacancy_descriptions(specialty)
    if not descriptions:
        return _empty_figure(f"Нет описаний вакансий для: {specialty}")

    top_skills: list[SkillFrequency] = db.get_skill_frequencies(specialty, top_n=top_n)
    if not top_skills:
        return _empty_figure(f"Нет данных по навыкам для: {specialty}")

    skill_lemmas = [s.lemma for s in top_skills]
    n = len(skill_lemmas)

    # Build co-occurrence matrix via NLPPipeline
    pipeline = NLPPipeline()
    cooc = np.zeros((n, n), dtype=int)

    for desc in descriptions:
        keywords = pipeline.extract_keywords(desc)
        present: set[str] = set()
        for lemma, _pos in keywords:
            if lemma in skill_lemmas:
                present.add(lemma)

        if len(present) >= 2:
            indices = [skill_lemmas.index(lemma) for lemma in present if lemma in skill_lemmas]
            for i in indices:
                for j in indices:
                    cooc[i, j] += 1

    # Build annotation text
    annot_text = [[str(v) if v > 0 else "" for v in row] for row in cooc]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=cooc.tolist(),
            x=skill_lemmas,
            y=skill_lemmas,
            colorscale="YlOrRd",
            text=annot_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> + <b>%{y}</b><br>Совместно: %{z} вакансий<extra></extra>",
            colorbar={"title": "Совместная<br>встречаемость"},
        )
    )

    fig.update_layout(
        title={
            "text": f"Совместная встречаемость навыков: {specialty}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        width=800 + n * 20,
        height=600 + n * 15,
        xaxis={"tickangle": 45},
        yaxis={"autorange": "reversed"},
        margin={"t": 60, "b": 120, "l": 120, "r": 40},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def generate_overlap_heatmap(
    db: Database,
    specialties: list[str],
    output_path: str | None = None,
) -> go.Figure:
    """Generate a heatmap showing Jaccard overlap scores between specialties.

    Uses SkillComparator.overlap_matrix() to compute pairwise Jaccard
    similarities between specialty skill sets.

    Args:
        db: Database instance.
        specialties: List of specialty keys to compare.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with a square go.Heatmap (specialties on both axes).
    """
    if len(specialties) < 2:
        return _empty_figure("Нужно минимум 2 специальности для матрицы пересечений")

    comparator = SkillComparator(db)
    matrix = comparator.overlap_matrix(specialties)

    # Build z-matrix and text annotations
    n = len(specialties)
    z_data: list[list[float]] = []
    annot_text: list[list[str]] = []

    for row_spec in specialties:
        row_data: list[float] = []
        row_text: list[str] = []
        for col_spec in specialties:
            val = matrix.get(row_spec, {}).get(col_spec, 0.0)
            row_data.append(val)
            row_text.append(f"{val:.2f}")
        z_data.append(row_data)
        annot_text.append(row_text)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=specialties,
            y=specialties,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            text=annot_text,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Jaccard: %{z:.3f}<extra></extra>",
            colorbar={"title": "Jaccard"},
        )
    )

    fig.update_layout(
        title={
            "text": "Пересечение навыков между специальностями",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        width=600 + n * 80,
        height=500 + n * 60,
        xaxis={"tickangle": 45},
        yaxis={"autorange": "reversed"},
        margin={"t": 60, "b": 120, "l": 120, "r": 40},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig
