"""Curriculum optimization via greedy skill selection and Pareto analysis.

Given a job market (vacancy descriptions for a specialty), the optimizer
finds minimal skill sets that achieve a target coverage level.  This supports
curriculum design decisions: "Which 20 skills give us 80% market coverage?"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hh_competency.analysis.skills import NoDataError, SkillAnalyzer
from hh_competency.simulation.monte_carlo import MonteCarloEngine
from hh_competency.storage.models import SimulationResult

if TYPE_CHECKING:
    from hh_competency.nlp.pipeline import NLPPipeline
    from hh_competency.storage.db import Database


class CurriculumOptimizer:
    """Greedy optimization of curriculum skill sets for market coverage.

    Works with pre-extracted vacancy skill-sets for fast coverage evaluation.
    Uses ``MonteCarloEngine`` internally for statistical fit estimation when
    confidence intervals are needed.

    The pipeline parameter is received externally — NLPPipeline is expensive
    to initialize (pymorphy3), so the caller manages its lifecycle.
    """

    def __init__(self, db: Database, pipeline: NLPPipeline) -> None:
        self._db = db
        self._pipeline = pipeline
        self._analyzer = SkillAnalyzer(db, pipeline)
        self._mc = MonteCarloEngine(db, pipeline)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def greedy_optimize(
        self,
        specialty: str,
        max_skills: int = 30,
        threshold: float = 0.7,
    ) -> dict:
        """Greedy algorithm to find the minimal skill set covering the market.

        At each step, evaluates every candidate skill's marginal coverage
        gain and picks the highest.  Stops when *max_skills* are selected or
        coverage exceeds *threshold*.

        Args:
            specialty: Specialty key (e.g. ``"data_science"``).
            max_skills: Maximum number of skills to select.
            threshold: Stop if average Jaccard coverage reaches this level.

        Returns:
            Dict with:
            - ``selected_skills``: ordered list of selected skill lemmas.
            - ``coverage_history``: coverage after each skill addition.
            - ``final_coverage``: coverage after the last step.

        Raises:
            NoDataError: If no vacancy data exists for the specialty.
        """
        vacancy_skill_sets = self._pre_extract_vacancy_skills(specialty)
        if not vacancy_skill_sets:
            raise NoDataError(
                f"No vacancies with extractable skills found for specialty: {specialty}"
            )

        # Market skills ranked by frequency.
        market_skills = self._get_market_skill_lemmas(specialty, top_n=200)

        selected: list[str] = []
        selected_set: set[str] = set()
        remaining = [s for s in market_skills if s not in selected_set]
        coverage_history: list[float] = []

        current_coverage = self._compute_exact_coverage(
            selected_set, vacancy_skill_sets
        )
        coverage_history.append(current_coverage)

        for _step in range(max_skills):
            if current_coverage >= threshold:
                break

            best_skill: str | None = None
            best_gain: float = -1.0

            for candidate in remaining:
                candidate_set = selected_set | {candidate}
                candidate_coverage = self._compute_exact_coverage(
                    candidate_set, vacancy_skill_sets
                )
                gain = candidate_coverage - current_coverage
                if gain > best_gain:
                    best_gain = gain
                    best_skill = candidate

            if best_skill is None or best_gain <= 0:
                break

            selected.append(best_skill)
            selected_set.add(best_skill)
            remaining.remove(best_skill)
            current_coverage += best_gain
            coverage_history.append(current_coverage)

        return {
            "selected_skills": selected,
            "coverage_history": coverage_history,
            "final_coverage": coverage_history[-1],
        }

    def pareto_frontier(
        self,
        specialty: str,
        max_skills: int = 50,
    ) -> list[dict]:
        """Trade-off analysis: skill count vs coverage.

        Runs greedy optimization for k = 1..*max_skills* and returns
        coverage at each k.  The resulting data can be plotted as a
        Pareto frontier — showing diminishing returns of adding skills.

        Args:
            specialty: Specialty key.
            max_skills: Maximum skills to evaluate up to.

        Returns:
            List of ``{"k": int, "coverage": float}`` dicts, one per
            skill count.  Coverage is monotonic non-decreasing.
        """
        vacancy_skill_sets = self._pre_extract_vacancy_skills(specialty)
        if not vacancy_skill_sets:
            return []

        market_skills = self._get_market_skill_lemmas(specialty, top_n=max_skills)
        selected_set: set[str] = set()
        frontier: list[dict] = []

        current_coverage = 0.0
        for k in range(1, max_skills + 1):
            if k > len(market_skills):
                break

            best_skill: str | None = None
            best_gain: float = -1.0

            for candidate in market_skills:
                if candidate in selected_set:
                    continue
                candidate_set = selected_set | {candidate}
                c = self._compute_exact_coverage(candidate_set, vacancy_skill_sets)
                gain = c - current_coverage
                if gain > best_gain:
                    best_gain = gain
                    best_skill = candidate

            if best_skill is None or best_gain <= 0:
                # No more gains — pad with final coverage.
                frontier.append({"k": k, "coverage": current_coverage})
                continue

            selected_set.add(best_skill)
            current_coverage += best_gain
            frontier.append({"k": k, "coverage": round(current_coverage, 6)})

        return frontier

    def what_if_add_skill(
        self,
        curriculum_skills: list[str],
        candidate_skills: list[str],
        specialty: str,
    ) -> list[dict]:
        """Evaluate the impact of adding each candidate skill to a curriculum.

        For each candidate, computes how much coverage improves if the skill
        is added.  Results are sorted by impact descending.

        Args:
            curriculum_skills: Baseline curriculum skill lemmas.
            candidate_skills: Skills to evaluate for addition.
            specialty: Specialty key.

        Returns:
            List of dicts sorted by ``delta`` descending:
            ``{"skill": str, "baseline_coverage": float,
              "new_coverage": float, "delta": float}``

        Raises:
            NoDataError: If no vacancy data exists for the specialty.
        """
        vacancy_skill_sets = self._pre_extract_vacancy_skills(specialty)
        if not vacancy_skill_sets:
            raise NoDataError(
                f"No vacancies with extractable skills found for specialty: {specialty}"
            )

        baseline_set = {s.lower() for s in curriculum_skills}
        baseline_coverage = self._compute_exact_coverage(
            baseline_set, vacancy_skill_sets
        )

        results: list[dict] = []
        for skill in candidate_skills:
            skill_norm = skill.lower()
            if skill_norm in baseline_set:
                results.append({
                    "skill": skill,
                    "baseline_coverage": round(baseline_coverage, 6),
                    "new_coverage": round(baseline_coverage, 6),
                    "delta": 0.0,
                })
                continue

            enhanced_set = baseline_set | {skill_norm}
            new_coverage = self._compute_exact_coverage(
                enhanced_set, vacancy_skill_sets
            )
            results.append({
                "skill": skill,
                "baseline_coverage": round(baseline_coverage, 6),
                "new_coverage": round(new_coverage, 6),
                "delta": round(new_coverage - baseline_coverage, 6),
            })

        results.sort(key=lambda r: r["delta"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Monte Carlo wrappers (for statistical confidence)
    # ------------------------------------------------------------------

    def mc_curriculum_fit(
        self,
        curriculum_skills: list[str],
        specialty: str,
        n_iter: int = 1000,
        sample_size: int | None = None,
    ) -> SimulationResult:
        """Estimate curriculum fit with confidence intervals via Monte Carlo.

        Thin wrapper around ``MonteCarloEngine.simulate_curriculum_fit``.

        Returns:
            ``SimulationResult`` with bootstrap confidence intervals.
        """
        return self._mc.simulate_curriculum_fit(
            curriculum_skills=curriculum_skills,
            specialty=specialty,
            n_iter=n_iter,
            sample_size=sample_size,
        )

    def mc_skill_impact(
        self,
        curriculum_skills: list[str],
        new_skill: str,
        specialty: str,
        n_iter: int = 500,
    ) -> dict[str, float]:
        """Estimate skill impact with paired Monte Carlo simulation.

        Thin wrapper around ``MonteCarloEngine.simulate_skill_impact``.
        """
        return self._mc.simulate_skill_impact(
            curriculum_skills=curriculum_skills,
            new_skill=new_skill,
            specialty=specialty,
            n_iter=n_iter,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pre_extract_vacancy_skills(self, specialty: str) -> list[set[str]]:
        """Extract skill lemma-sets from all vacancy descriptions once.

        Filters out vacancies with no extractable skills.
        """
        descriptions = self._db.get_vacancy_descriptions(specialty)
        skill_sets: list[set[str]] = []
        for desc in descriptions:
            keywords = self._pipeline.extract_keywords(desc)
            lemmas = {lemma.lower() for lemma, _pos in keywords}
            if lemmas:
                skill_sets.append(lemmas)
        return skill_sets

    def _get_market_skill_lemmas(
        self, specialty: str, top_n: int = 200
    ) -> list[str]:
        """Retrieve top market skill lemmas sorted by frequency descending.

        Falls back to compute_frequencies if nothing is cached.
        """
        skills = self._analyzer.top_skills(specialty, n=top_n)
        return [f.lemma.lower() for f in skills]

    @staticmethod
    def _compute_exact_coverage(
        skill_set: set[str],
        vacancy_skill_sets: list[set[str]],
    ) -> float:
        """Compute exact mean Jaccard coverage over all vacancies.

        Deterministic — no sampling.  Used by the greedy optimizer for
        fast, reproducible coverage evaluation.

        Args:
            skill_set: Curriculum skill lemmas (lowercased).
            vacancy_skill_sets: Pre-extracted vacancy lemma-sets.

        Returns:
            Mean Jaccard similarity across all vacancies.
        """
        if not vacancy_skill_sets:
            return 0.0
        total = 0.0
        for vs in vacancy_skill_sets:
            intersection = len(skill_set & vs)
            union = len(skill_set | vs)
            if union > 0:
                total += intersection / union
        return total / len(vacancy_skill_sets)
