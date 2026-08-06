"""Salary distribution charts: box plots, histograms, and skill-vs-salary analysis.

Uses Plotly go.Box, go.Histogram, and go.Bar to visualize salary distributions
from HH.ru vacancy data across specialties and skill levels.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from hh_competency.nlp.pipeline import NLPPipeline
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


def _compute_salary_avg(df: pd.DataFrame) -> pd.Series:
    """Compute average salary from 'salary_from' and 'salary_to' columns.

    Uses (from + to) / 2 when both are present, uses the single value when
    only one is available, and drops rows where both are NaN.
    """
    from_vals = df.get("salary_from", pd.Series(dtype=float)).fillna(0)
    to_vals = df.get("salary_to", pd.Series(dtype=float)).fillna(0)

    # Where 'to' is present, average; otherwise use 'from'
    avg = (from_vals + to_vals) / 2.0

    # If 'to' was NaN (now 0), use 'from' only
    has_to = df.get("salary_to", pd.Series(dtype=float)).notna()
    avg[~has_to] = from_vals[~has_to]

    # Drop zeros (no salary data at all)
    return avg


def generate_salary_boxplot(
    db: Database,
    specialties: list[str],
    output_path: str | None = None,
) -> go.Figure:
    """Generate a box plot of salary distributions across specialties.

    Each box represents the distribution of average salaries (mean of from/to)
    for vacancies in a given specialty.

    Args:
        db: Database instance.
        specialties: List of specialty keys.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with go.Box traces per specialty.
    """
    if not specialties:
        return _empty_figure("Не указаны специальности для анализа зарплат")

    fig = go.Figure()
    has_data = False

    for idx, spec in enumerate(specialties):
        df = db.get_salary_data(spec)
        if df.empty:
            continue

        avg_salary = _compute_salary_avg(df)
        # Filter out zero salaries (no data)
        avg_salary = avg_salary[avg_salary > 0]

        if avg_salary.empty:
            continue

        has_data = True
        color = _COLORS[idx % len(_COLORS)]

        fig.add_trace(
            go.Box(
                y=avg_salary,
                name=spec,
                marker={"color": color},
                boxmean="sd",
                hovertemplate="<b>%{data.name}</b><br>"
                "Медиана: %{median:,.0f} ₽<br>"
                "Q1: %{q1:,.0f} ₽<br>"
                "Q3: %{q3:,.0f} ₽<br>"
                "Мин: %{lowerfence:,.0f} ₽<br>"
                "Макс: %{upperfence:,.0f} ₽<extra></extra>",
            )
        )

    if not has_data:
        return _empty_figure("Нет данных о зарплатах ни для одной специальности")

    fig.update_layout(
        title={
            "text": "Распределение зарплат по специальностям",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        yaxis_title="Зарплата (RUB)",
        xaxis_title="Специальность",
        width=1000,
        height=600,
        margin={"t": 60, "b": 120, "l": 100, "r": 40},
        xaxis={"tickangle": 30},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def generate_salary_histogram(
    db: Database,
    specialty: str,
    by_experience: bool = True,
    output_path: str | None = None,
) -> go.Figure:
    """Generate a histogram of salary distribution for a specialty.

    Optionally splits data into separate traces by experience level.

    Args:
        db: Database instance.
        specialty: Specialty key.
        by_experience: If True, create separate histogram traces for each
                        experience level found in the data.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with histogram traces.
    """
    df = db.get_salary_data(specialty)
    if df.empty:
        return _empty_figure(f"Нет данных о зарплатах для: {specialty}")

    avg_salary = _compute_salary_avg(df)
    avg_salary = avg_salary[avg_salary > 0]

    if avg_salary.empty:
        return _empty_figure(f"Нет данных о зарплатах (после фильтрации) для: {specialty}")

    fig = go.Figure()

    if by_experience and "experience" in df.columns:
        exp_col = df["experience"]
        exp_levels = exp_col.dropna().unique()

        if len(exp_levels) <= 1:
            # Single experience level — fall back to single histogram
            by_experience = False

    if not by_experience:
        fig.add_trace(
            go.Histogram(
                x=avg_salary,
                nbinsx=30,
                marker={"color": _COLORS[0], "line": {"color": "white", "width": 1}},
                hovertemplate="Зарплата: %{x:,.0f} ₽<br>Количество: %{y}<extra></extra>",
                name="Все уровни",
            )
        )
    else:
        exp_col = df["experience"]
        exp_levels_sorted = sorted(
            exp_col.dropna().unique(), key=lambda x: str(x)
        )
        for idx, level in enumerate(exp_levels_sorted):
            mask = exp_col == level
            subset = avg_salary[mask.values]
            if subset.empty:
                continue
            color = _COLORS[idx % len(_COLORS)]
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    nbinsx=30,
                    name=str(level),
                    marker={"color": color, "line": {"color": "white", "width": 1}},
                    opacity=0.7,
                    hovertemplate=(
                        "<b>%{data.name}</b><br>"
                        "Зарплата: %{x:,.0f} ₽<br>"
                        "Количество: %{y}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title={
            "text": f"Распределение зарплат: {specialty}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="Зарплата (RUB)",
        yaxis_title="Количество вакансий",
        barmode="overlay",
        width=1000,
        height=600,
        margin={"t": 60, "b": 60, "l": 80, "r": 40},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig


def generate_salary_vs_skills(
    db: Database,
    specialty: str,
    top_skills: list[str],
    output_path: str | None = None,
) -> go.Figure:
    """Generate a grouped bar chart: median salary for vacancies with vs without each skill.

    For each skill in top_skills, computes the median salary of vacancies that
    mention the skill and those that do not, showing the salary premium.

    Args:
        db: Database instance.
        specialty: Specialty key.
        top_skills: List of skill lemmas to analyze.
        output_path: If provided, save self-contained HTML.

    Returns:
        Plotly Figure with grouped bars comparing median salaries.
    """
    if not top_skills:
        return _empty_figure("Не указаны навыки для анализа")

    # Query descriptions and salaries together via raw connection
    try:
        df = db.connection.execute(
            """
            SELECT
                json_extract_string(rv.data, '$.description') AS description,
                json_extract_string(rv.data, '$.salary.from')::FLOAT AS salary_from,
                json_extract_string(rv.data, '$.salary.to')::FLOAT AS salary_to
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
              AND json_extract_string(rv.data, '$.description') IS NOT NULL
              AND json_extract_string(rv.data, '$.salary.from') IS NOT NULL
            """,
            [specialty],
        ).df()
    except Exception:
        return _empty_figure(f"Ошибка при запросе данных для: {specialty}")

    if df.empty:
        return _empty_figure(f"Нет данных о зарплатах и описаниях для: {specialty}")

    # Compute average salary
    df["salary_avg"] = _compute_salary_avg(df)
    df = df[df["salary_avg"] > 0]

    if df.empty:
        return _empty_figure(f"Нет валидных данных о зарплатах для: {specialty}")

    # Check which descriptions contain each skill
    pipeline = NLPPipeline()
    skill_presence: dict[str, list[bool]] = {}

    for desc in df["description"]:
        keywords = pipeline.extract_keywords(str(desc) if desc else "")
        lemmas_present = {lemma for lemma, _pos in keywords}
        for skill in top_skills:
            if skill not in skill_presence:
                skill_presence[skill] = []
            skill_presence[skill].append(skill in lemmas_present)

    # Compute median salary for with/without each skill
    with_medians: list[float] = []
    without_medians: list[float] = []
    skill_labels: list[str] = []

    for skill in top_skills:
        presence = skill_presence.get(skill, [False] * len(df))
        mask = pd.Series(presence, index=df.index)
        with_skill = df.loc[mask, "salary_avg"]
        without_skill = df.loc[~mask, "salary_avg"]

        if with_skill.empty and without_skill.empty:
            continue

        skill_labels.append(skill)
        with_medians.append(float(with_skill.median()) if not with_skill.empty else 0.0)
        without_medians.append(
            float(without_skill.median()) if not without_skill.empty else 0.0
        )

    if not skill_labels:
        return _empty_figure(f"Ни один из указанных навыков не найден в вакансиях: {specialty}")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="С навыком",
            x=skill_labels,
            y=with_medians,
            marker={"color": _COLORS[0]},
            hovertemplate="<b>%{x}</b><br>С навыком: %{y:,.0f} ₽<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Без навыка",
            x=skill_labels,
            y=without_medians,
            marker={"color": _COLORS[3]},
            hovertemplate="<b>%{x}</b><br>Без навыка: %{y:,.0f} ₽<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"Зарплата vs наличие навыка: {specialty}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="Навык",
        yaxis_title="Медианная зарплата (RUB)",
        barmode="group",
        width=1000,
        height=600,
        margin={"t": 60, "b": 120, "l": 100, "r": 40},
        xaxis={"tickangle": 45},
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return fig
