"""Competency abstraction: transform raw skill clusters into abstract competency characteristics.

Uses AxisDiscoveryEngine for statistical axis discovery, then names each
axis using heuristic category rules and computes per-axis vacancy coverage.
"""

from __future__ import annotations

from collections import Counter

from msgspec import Struct

from hh_competency.analysis.axis_discovery import (
    AxisDiscoveryEngine,
    _classify_skill_to_category,
    _name_axis_from_skills,
)
from hh_competency.storage.db import Database

# ────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────


class Characteristic(Struct):
    """A single abstract competency characteristic derived from data."""

    name: str  # abstract label e.g. "Экспериментальные методы"
    category: str  # high-level category from naming table
    representative_skills: list[str]  # top 5 defining skills
    coverage_pct: float  # % of vacancies containing ≥1 skill from this characteristic
    stability_score: float  # bootstrap ARI of the clustering scheme
    is_core: bool  # coverage > 25% threshold


class CompetencyAbstraction(Struct):
    """Discovery result: 3–9 abstract competency characteristics."""

    specialty: str
    characteristics: list[Characteristic]
    optimal_n: int  # the data-optimized number (3–9)
    verdict: str  # human-readable summary


# ────────────────────────────────────────────────────────
# Main engine
# ────────────────────────────────────────────────────────


class CompetencyAbstractor:
    """Transform raw skill tokens into 3–9 abstract competency characteristics.

    Pipeline:
    1. Retrieve top-N skills from DB for the specialty.
    2. Delegate to AxisDiscoveryEngine to evaluate k=2..9 schemes.
    3. Select the scheme with the best harmonic-mean combined score.
    4. Name each axis using heuristic category rules (not raw skills).
    5. Compute per-axis vacancy coverage and filter low-quality axes.
    6. Clamp to [min_chars, max_chars] range.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._engine = AxisDiscoveryEngine(db)

    # ── Entry point ──────────────────────────────────

    def discover(
        self,
        specialty: str,
        min_chars: int = 3,
        max_chars: int = 9,
        top_n_skills: int = 80,
        n_bootstrap: int = 200,
        random_seed: int = 42,
    ) -> CompetencyAbstraction:
        """Discover 3–9 abstract competency characteristics from skill co-occurrence.

        Args:
            specialty: Specialty key (e.g. 'physics').
            min_chars: Minimum number of characteristics to return.
            max_chars: Maximum number of characteristics to return.
            top_n_skills: Top N skills fed into axis discovery.
            n_bootstrap: Bootstrap iterations for stability estimation.
            random_seed: Random seed for reproducibility.

        Returns:
            CompetencyAbstraction with 3–9 data-driven characteristics.
        """
        skills = self._db.get_skill_frequencies(specialty, top_n=top_n_skills)
        if not skills:
            return CompetencyAbstraction(
                specialty=specialty,
                characteristics=[],
                optimal_n=0,
                verdict=f"Нет навыков для специальности: {specialty}",
            )

        n_vacancies = self._db.get_vacancy_count(specialty)

        # Run axis discovery for k=2..9
        result = self._engine.discover(
            specialty=specialty,
            skills=skills,
            n_vacancies=n_vacancies,
            min_k=2,
            max_k=9,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        if not result.comparison.schemes:
            return CompetencyAbstraction(
                specialty=specialty,
                characteristics=[],
                optimal_n=0,
                verdict=f"Недостаточно данных для выделения характеристик: {specialty}",
            )

        # Select scheme with best harmonic-mean combined score
        best_scheme = max(result.comparison.schemes, key=lambda s: s.combined)

        # Get vacancy descriptions for per-axis coverage computation
        descriptions = self._db.get_vacancy_descriptions(specialty)

        # Build characteristics from axes
        characteristics = _axes_to_characteristics(
            best_scheme.axes,
            best_scheme.stability,
            descriptions,
            coverage_threshold=0.05,
        )

        # Clamp to [min_chars, max_chars]
        characteristics = _clamp_characteristics(
            characteristics,
            best_scheme.axes,
            best_scheme.stability,
            descriptions,
            min_chars,
            max_chars,
        )

        optimal_n = len(characteristics)

        # Build verdict
        core_count = sum(1 for c in characteristics if c.is_core)
        categories = sorted({c.category for c in characteristics})
        verdict = (
            f"{specialty}: обнаружены {optimal_n} "
            f"компетентностных характеристик ({core_count} ядерных) "
            f"в категориях: {', '.join(categories)}"
        )

        return CompetencyAbstraction(
            specialty=specialty,
            characteristics=characteristics,
            optimal_n=optimal_n,
            verdict=verdict,
        )


# ────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────


def _compute_axis_coverage(axis_skills: list[str], descriptions: list[str]) -> float:
    """Fraction of vacancies containing at least one skill from this axis."""
    if not descriptions:
        return 0.0
    count = 0
    for desc in descriptions:
        desc_lower = desc.lower()
        if any(s.lower() in desc_lower for s in axis_skills):
            count += 1
    return count / len(descriptions)


def _dominant_category(skills: list[str]) -> str:
    """Find the most common heuristic category among a set of skills."""
    cats = Counter()
    for skill in skills:
        cat = _classify_skill_to_category(skill)
        if cat:
            cats[cat] += 1
    if cats:
        return cats.most_common(1)[0][0]
    return "Неопределённая категория"


def _axes_to_characteristics(
    axes: list,
    scheme_stability: float,
    descriptions: list[str],
    coverage_threshold: float,
) -> list[Characteristic]:
    """Convert axis definitions to filtered Characteristic objects."""
    characteristics: list[Characteristic] = []
    for axis in axes:
        coverage = _compute_axis_coverage(axis.skills, descriptions)
        stability = scheme_stability

        # Drop low-quality axes
        if coverage < coverage_threshold or stability < -1.0:
            continue

        category = _dominant_category(axis.defining)
        name = _name_axis_from_skills(axis.defining)

        characteristics.append(
            Characteristic(
                name=name,
                category=category,
                representative_skills=list(axis.defining[:5]),
                coverage_pct=round(coverage * 100, 1),
                stability_score=round(stability, 3),
                is_core=coverage >= 0.25,
            )
        )
    return characteristics


def _clamp_characteristics(
    filtered: list[Characteristic],
    all_axes: list,
    scheme_stability: float,
    descriptions: list[str],
    min_chars: int,
    max_chars: int,
) -> list[Characteristic]:
    """Ensure characteristics count is within [min_chars, max_chars]."""
    if len(filtered) > max_chars:
        # Keep top by coverage
        filtered.sort(key=lambda c: c.coverage_pct, reverse=True)
        return filtered[:max_chars]

    if len(filtered) < min_chars:
        # Relax coverage threshold to fill up to min_chars
        seen_names = {c.name for c in filtered}
        relaxed: list[Characteristic] = []
        for axis in all_axes:
            name = _name_axis_from_skills(axis.defining)
            if name in seen_names:
                continue
            coverage = _compute_axis_coverage(axis.skills, descriptions)
            if coverage < 0.02:
                continue
            category = _dominant_category(axis.defining)
            relaxed.append(
                Characteristic(
                    name=name,
                    category=category,
                    representative_skills=list(axis.defining[:5]),
                    coverage_pct=round(coverage * 100, 1),
                    stability_score=round(scheme_stability, 3),
                    is_core=coverage >= 0.25,
                )
            )
            seen_names.add(name)
            if len(filtered) + len(relaxed) >= max_chars:
                break

        filtered = filtered + relaxed
        filtered.sort(key=lambda c: c.coverage_pct, reverse=True)

    return filtered


# ────────────────────────────────────────────────────────
# Convenience API
# ────────────────────────────────────────────────────────


def abstract_characteristics(db: Database, specialty: str) -> CompetencyAbstraction:
    """One-shot discovery of abstract competency characteristics.

    Args:
        db: Database connection.
        specialty: Specialty key (e.g. 'physics').

    Returns:
        CompetencyAbstraction with 3–9 data-driven characteristics.
    """
    abstractor = CompetencyAbstractor(db)
    return abstractor.discover(specialty)


def characteristics_to_markdown(abstraction: CompetencyAbstraction) -> str:
    """Render a CompetencyAbstraction as a markdown report.

    Args:
        abstraction: Discovery result from CompetencyAbstractor.

    Returns:
        Markdown-formatted string suitable for reporting.
    """
    lines: list[str] = [
        f"# Характеристики компетенций: {abstraction.specialty}",
        "",
        f"**Вердикт:** {abstraction.verdict}",
        "",
        f"**Оптимальное число характеристик:** {abstraction.optimal_n}",
        "",
    ]

    for i, ch in enumerate(abstraction.characteristics, 1):
        core_marker = " ★ ЯДЕРНАЯ" if ch.is_core else ""
        lines.extend(
            [
                f"## {i}. {ch.name}{core_marker}",
                "",
                f"- **Категория:** {ch.category}",
                f"- **Покрытие вакансий:** {ch.coverage_pct}%",
                f"- **Стабильность (bootstrap ARI):** {ch.stability_score}",
                f"- **Ключевые навыки:** {', '.join(ch.representative_skills)}",
                "",
            ]
        )

    return "\n".join(lines)
