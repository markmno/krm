"""Visualization module: Plotly charts for competency-role model analysis.

Public API:
    radar.py  — Spider/radar charts comparing skill profiles across specialties
    bars.py   — Horizontal bar charts for skills and role profiles
    heatmap.py — Co-occurrence and overlap heatmaps
    salary.py — Salary distribution box plots, histograms, skills-vs-salary
"""

from hh_competency.viz.bars import (
    generate_comparison_bar_chart,
    generate_role_profile_chart,
    generate_skill_bar_chart,
)
from hh_competency.viz.heatmap import (
    generate_cooccurrence_heatmap,
    generate_overlap_heatmap,
)
from hh_competency.viz.radar import generate_radar_chart, radar_data_json
from hh_competency.viz.salary import (
    generate_salary_boxplot,
    generate_salary_histogram,
    generate_salary_vs_skills,
)

__all__ = [
    # radar
    "generate_radar_chart",
    "radar_data_json",
    # bars
    "generate_skill_bar_chart",
    "generate_comparison_bar_chart",
    "generate_role_profile_chart",
    # heatmap
    "generate_cooccurrence_heatmap",
    "generate_overlap_heatmap",
    # salary
    "generate_salary_boxplot",
    "generate_salary_histogram",
    "generate_salary_vs_skills",
]
