"""Yearly dynamics analysis: trend detection, emerging skills, cluster stability over time.

Proves that discovered competency patterns are sustainable — not one-year anomalies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hh_competency.storage.db import Database


def _mann_kendall(values: list[float]) -> tuple[float, float, str]:
    """Mann-Kendall trend test (non-parametric).

    Args:
        values: Time-ordered sequence of values (e.g. yearly frequencies).

    Returns:
        (tau, p_value, direction): Kendall's tau, approximate p-value, and
        trend direction: 'growing', 'declining', or 'stable'.
    """
    import math

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

    tau = s / (n * (n - 1) / 2) if n > 1 else 0.0

    variance = n * (n - 1) * (2 * n + 5) / 18.0
    if variance == 0:
        return tau, 1.0, "stable"

    z = s / (variance**0.5)
    p_value = 2 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    direction = "stable"
    if p_value < 0.05 or abs(tau) > 0.3:
        direction = "growing" if tau > 0 else "declining"

    return tau, p_value, direction


def _linear_slope(years: list[int], values: list[float]) -> float:
    """Simple linear regression slope. Positive = growing."""
    n = len(years)
    if n < 2:
        return 0.0
    mean_yr = sum(years) / n
    mean_val = sum(values) / n
    num = sum((years[i] - mean_yr) * (values[i] - mean_val) for i in range(n))
    den = sum((years[i] - mean_yr) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def analyze_trends(
    db: Database,
    specialty: str,
    min_count: int = 2,
    top_n: int = 30,
) -> dict:
    """Full yearly trend analysis for a specialty.

    Returns:
        Dict with available_years, skills (with tau/p-value/direction/slope),
        emerging/declining/stable classifications, and sustainability summary.
    """
    from collections import Counter

    from hh_competency.nlp.pipeline import NLPPipeline

    by_year = db.get_vacancies_by_year(specialty)
    if not by_year or len(by_year) < 2:
        return {
            "specialty": specialty,
            "available_years": sorted(by_year.keys()) if by_year else [],
            "total_vacancies": sum(len(v) for v in by_year.values()),
            "error": (
                "Нужны данные минимум за 2 года"
                if by_year
                else "Нет данных о вакансиях"
            ),
            "skills": [],
            "emerging_skills": [],
            "declining_skills": [],
            "stable_skills": [],
            "summary": "Недостаточно данных для анализа трендов — нужно минимум 2 года.",
        }

    years = sorted(by_year.keys())
    total_vacancies = sum(len(v) for v in by_year.values())

    pipeline = NLPPipeline()
    yearly_skills: dict[int, Counter[str]] = {}
    for yr in years:
        counter: Counter[str] = Counter()
        for desc in by_year[yr]:
            for lemma, _pos in pipeline.extract_keywords(desc):
                counter[lemma] += 1
        yearly_skills[yr] = counter

    all_skills: set[str] = set()
    for yr in years:
        for lemma, cnt in yearly_skills[yr].items():
            if cnt >= min_count:
                all_skills.add(lemma)

    if not all_skills:
        return {
            "specialty": specialty,
            "available_years": years,
            "total_vacancies": total_vacancies,
            "skills": [],
            "emerging_skills": [],
            "declining_skills": [],
            "stable_skills": [],
            "summary": f"Недостаточно навыков при min_count={min_count}",
        }

    skill_trends: list[dict] = []
    for lemma in sorted(all_skills):
        values = [yearly_skills[yr].get(lemma, 0) for yr in years]
        if sum(values) < min_count:
            continue
        tau, p, direction = _mann_kendall([float(v) for v in values])
        slope = _linear_slope(years, [float(v) for v in values])
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
        s for s in skill_trends
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
                f"Ролевые модели не являются артефактом одного года."
            )
        elif stable_pct > 30:
            summary = (
                f"Паттерны ЧАСТИЧНО УСТОЙЧИВЫ: {stable_pct:.0f}% навыков стабильны, "
                f"{growing_pct:.0f}% растут. Рынок эволюционирует, но ядро сохраняется."
            )
        else:
            summary = (
                f"Паттерны НЕСТАБИЛЬНЫ: только {stable_pct:.0f}% навыков стабильны. "
                f"Рынок в фазе активной трансформации. "
                f"Рекомендуется ежегодный пересмотр модели."
            )

    return {
        "specialty": specialty,
        "available_years": years,
        "total_vacancies": total_vacancies,
        "skills": skill_trends[:top_n],
        "total_analyzed_skills": len(skill_trends),
        "emerging_skills": emerging[:top_n],
        "declining_skills": declining[:top_n],
        "stable_skills": stable[:top_n],
        "newly_emerged_skills": newly_emerged[:15],
        "summary": summary,
        "stability_pct": round(stable_pct if skill_trends else 0, 1),
    }


def analyze_salary_trends(db: Database, specialty: str) -> dict:
    """Yearly salary trend analysis.

    Returns:
        Dict with yearly median/mean/P25/P75 salary and slope.
    """
    salary_by_year = db.get_yearly_salary_data(specialty)
    if not salary_by_year or len(salary_by_year) < 2:
        return {
            "specialty": specialty,
            "error": "Нужны данные о зарплатах минимум за 2 года",
            "trends": [],
        }

    years = sorted(salary_by_year.keys())
    trends = []
    for yr in years:
        salaries = [r["from"] for r in salary_by_year[yr] if r["from"]]
        if not salaries:
            continue
        s_sorted = sorted(salaries)
        n = len(s_sorted)
        trends.append({
            "year": yr,
            "median": s_sorted[n // 2],
            "mean": round(sum(salaries) / n),
            "p25": s_sorted[n // 4],
            "p75": s_sorted[3 * n // 4],
            "vacancies_with_salary": n,
        })

    if trends:
        median_values = [t["median"] for t in trends]
        slope = _linear_slope([t["year"] for t in trends], [float(v) for v in median_values])
        tau, p, direction = _mann_kendall([float(v) for v in median_values])
        salary_verdict = (
            "Медианная зарплата "
            + ("растёт" if direction == "growing" else "падает" if direction == "declining" else "стабильна")
            + (f" (tau={tau:.2f}, p={p:.3f})" if p < 0.10 else " (статистически незначимо)")
        )
    else:
        slope = 0.0
        tau = 0.0
        p = 1.0
        direction = "unknown"
        salary_verdict = "Недостаточно данных"

    return {
        "specialty": specialty,
        "available_years": years,
        "trends": trends,
        "salary_slope_rub_year": round(slope, 0),
        "salary_tau": round(tau, 3),
        "salary_trend_direction": direction,
        "salary_trend_p_value": round(p, 4),
        "verdict": salary_verdict,
    }
