"""Yearly dynamics analysis: trend detection and skill evolution.

Provides Mann-Kendall trend test, linear regression slope,
and aggregate trend analysis for skill time series and salary data.
All inputs are pre-computed Python dicts/lists — no database dependency.
"""

from __future__ import annotations

import math
from collections import Counter

__all__ = [
    "mann_kendall",
    "linear_slope",
    "analyze_trends",
    "analyze_salary_trends",
]


# ---------------------------------------------------------------------------
# Private helpers — algorithmically identical to hh_competency originals
# ---------------------------------------------------------------------------


def _mann_kendall(values: list[float]) -> tuple[float, float, str]:
    """Mann-Kendall trend test (non-parametric).

    Args:
        values: Time-ordered sequence of values (e.g. yearly frequencies).

    Returns:
        (tau, p_value, direction): Kendall's tau, approximate p-value, and
        trend direction: 'growing', 'declining', or 'stable'.
    """
    n = len(values)
    if n < 3:
        return 0.0, 1.0, "stable"

    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    tau = s / (n * (n - 1) / 2)

    variance = n * (n - 1) * (2 * n + 5) / 18.0
    z = s / (variance ** 0.5)
    p_value = math.erfc(abs(z) / math.sqrt(2.0))

    direction = "stable"
    if p_value < 0.05 or abs(tau) > 0.3:
        direction = "growing" if tau > 0 else "declining"

    return tau, p_value, direction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mann_kendall(values: list[float]) -> dict[str, float | str]:
    """Mann-Kendall trend test — public wrapper returning a dict.

    Args:
        values: Time-ordered sequence of numeric observations.

    Returns:
        Dict with keys:
            tau (float): Kendall rank correlation coefficient (-1..1).
            p_value (float): Approximate two-sided p-value.
            direction (str): 'growing', 'declining', or 'stable'.
    """
    tau, p_value, direction = _mann_kendall(values)
    return {"tau": round(tau, 3), "p_value": round(p_value, 4), "direction": direction}


def linear_slope(years: list[int], values: list[float]) -> float:
    """Simple linear regression slope.

    Args:
        years: Year ordinals.
        values: Corresponding time series values.

    Returns:
        Slope (units per year). Positive = growing, negative = declining.
    """
    n = len(years)
    if n < 2:
        return 0.0
    mean_yr = sum(years) / n
    mean_val = sum(values) / n
    num = sum((years[i] - mean_yr) * (values[i] - mean_val) for i in range(n))
    den = sum((years[i] - mean_yr) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def analyze_trends(
    skill_time_series: dict[int, dict[str, int]],
    min_count: int = 2,
    top_n: int = 30,
) -> dict:
    """Full yearly trend analysis for a pre-computed skill time series.

    Args:
        skill_time_series: Mapping ``{year: {skill_lemma: count}}``,
            e.g. ``{2020: {"python": 45, "sql": 30}, 2021: {"python": 52, "sql": 28}}``.
        min_count: Minimum total occurrences for a skill to be analyzed.
        top_n: Maximum skills to include in each result category.

    Returns:
        Dict with keys:
            available_years (list[int]): Sorted years present in input.
            skills (list[dict]): Top-N skills with tau, p_value, direction, slope.
            total_analyzed_skills (int): Total number of skills that met min_count.
            emerging_skills (list[dict]): Skills classified as growing.
            declining_skills (list[dict]): Skills classified as declining.
            stable_skills (list[dict]): Skills classified as stable.
            newly_emerged_skills (list[dict]): Skills absent in earliest year
                but present in latest year (not already in emerging).
            summary (str): Human-readable verdict on pattern stability.
            stability_pct (float): Percentage of skills classified as stable.
    """
    if not skill_time_series or len(skill_time_series) < 2:
        return {
            "available_years": (
                sorted(skill_time_series.keys()) if skill_time_series else []
            ),
            "error": (
                "Нужны данные минимум за 2 года"
                if skill_time_series
                else "Нет данных"
            ),
            "skills": [],
            "total_analyzed_skills": 0,
            "emerging_skills": [],
            "declining_skills": [],
            "stable_skills": [],
            "newly_emerged_skills": [],
            "summary": "Недостаточно данных для анализа трендов — нужно минимум 2 года.",
            "stability_pct": 0.0,
        }

    years = sorted(skill_time_series.keys())
    yearly_skills: dict[int, Counter[str]] = {
        yr: Counter(skills) for yr, skills in skill_time_series.items()
    }

    all_skills: set[str] = set()
    for yr in years:
        for lemma, cnt in yearly_skills[yr].items():
            if cnt >= min_count:
                all_skills.add(lemma)

    if not all_skills:
        return {
            "available_years": years,
            "skills": [],
            "total_analyzed_skills": 0,
            "emerging_skills": [],
            "declining_skills": [],
            "stable_skills": [],
            "newly_emerged_skills": [],
            "summary": f"Недостаточно навыков при min_count={min_count}",
            "stability_pct": 0.0,
        }

    skill_trends: list[dict] = []
    for lemma in sorted(all_skills):
        values = [yearly_skills[yr].get(lemma, 0) for yr in years]
        if sum(values) < min_count:
            continue
        float_values = [float(v) for v in values]
        tau, p, direction = _mann_kendall(float_values)
        slope = linear_slope(years, float_values)
        skill_trends.append({
            "lemma": lemma,
            "years": {str(yr): yearly_skills[yr].get(lemma, 0) for yr in years},
            "tau": round(tau, 3),
            "trend_p_value": round(p, 4),
            "direction": direction,
            "slope": round(slope, 3),
            "total_frequency": sum(values),
        })

    skill_trends.sort(key=lambda x: x["total_frequency"], reverse=True)

    emerging = [s for s in skill_trends if s["direction"] == "growing"]
    declining = [s for s in skill_trends if s["direction"] == "declining"]
    stable = [s for s in skill_trends if s["direction"] == "stable"]

    latest_yr = years[-1]
    earliest_yr = years[0]
    newly_emerged = [
        s
        for s in skill_trends
        if s["years"].get(str(earliest_yr), 0) == 0
        and s["years"].get(str(latest_yr), 0) > 0
        and s not in emerging
    ]

    if not skill_trends:
        stable_pct = 0.0
        summary = "Недостаточно навыков для оценки устойчивости паттернов."
    else:
        stable_pct = len(stable) / len(skill_trends) * 100
        growing_pct = len(emerging) / len(skill_trends) * 100
        if stable_pct > 60:
            summary = (
                f"Паттерны УСТОЙЧИВЫ: {stable_pct:.0f}% навыков стабильны "
                f"на протяжении {years[0]}-{latest_yr} гг. "
                "Ролевые модели не являются артефактом одного года."
            )
        elif stable_pct > 30:
            summary = (
                f"Паттерны ЧАСТИЧНО УСТОЙЧИВЫ: {stable_pct:.0f}% навыков стабильны, "
                f"{growing_pct:.0f}% растут. Рынок эволюционирует, но ядро сохраняется."
            )
        else:
            summary = (
                f"Паттерны НЕСТАБИЛЬНЫ: только {stable_pct:.0f}% навыков стабильны. "
                "Рынок в фазе активной трансформации. "
                "Рекомендуется ежегодный пересмотр модели."
            )

    return {
        "available_years": years,
        "skills": skill_trends[:top_n],
        "total_analyzed_skills": len(skill_trends),
        "emerging_skills": emerging[:top_n],
        "declining_skills": declining[:top_n],
        "stable_skills": stable[:top_n],
        "newly_emerged_skills": newly_emerged[:15],
        "summary": summary,
        "stability_pct": round(stable_pct if skill_trends else 0, 1),
    }


def analyze_salary_trends(
    salary_records: list[dict[str, int | float]],
) -> dict:
    """Yearly salary trend analysis from pre-computed statistics.

    Args:
        salary_records: List of per-year dicts, each containing::

            {
                "year": 2022,
                "p25": 120000,
                "p50": 180000,
                "p75": 250000,
            }

    Returns:
        Dict with keys:
            available_years (list[int]): Sorted years.
            trends (list[dict]): Per-year statistics (year, median, p25, p75).
            salary_slope_rub_year (float): Linear regression slope of median.
            salary_tau (float): Mann-Kendall tau for median series.
            salary_trend_direction (str): 'growing', 'declining', 'stable', or 'unknown'.
            salary_trend_p_value (float): Mann-Kendall p-value.
            verdict (str): Human-readable interpretation.
    """
    if not salary_records or len(salary_records) < 2:
        return {
            "available_years": (
                [rec["year"] for rec in salary_records] if salary_records else []
            ),
            "error": "Нужны данные о зарплатах минимум за 2 года",
            "trends": [],
            "salary_slope_rub_year": 0.0,
            "salary_tau": 0.0,
            "salary_trend_direction": "unknown",
            "salary_trend_p_value": 1.0,
            "verdict": "Недостаточно данных",
        }

    years: list[int] = [int(rec["year"]) for rec in salary_records]
    trends: list[dict] = [
        {
            "year": rec["year"],
            "median": rec["p50"],
            "p25": rec["p25"],
            "p75": rec["p75"],
        }
        for rec in salary_records
    ]

    if trends:
        median_values = [float(t["median"]) for t in trends]
        slope = linear_slope(years, median_values)
        tau, p, direction = _mann_kendall(median_values)
        salary_verdict = (
            "Медианная зарплата "
            + {"growing": "растёт", "declining": "падает"}.get(direction, "стабильна")
            + (
                f" (tau={tau:.2f}, p={p:.3f})"
                if p < 0.10
                else " (статистически незначимо)"
            )
        )
    else:
        slope = 0.0
        tau = 0.0
        p = 1.0
        direction = "unknown"
        salary_verdict = "Недостаточно данных"

    return {
        "available_years": years,
        "trends": trends,
        "salary_slope_rub_year": round(slope, 0),
        "salary_tau": round(tau, 3),
        "salary_trend_direction": direction,
        "salary_trend_p_value": round(p, 4),
        "verdict": salary_verdict,
    }
