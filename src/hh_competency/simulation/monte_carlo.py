"""Monte Carlo simulation for curriculum-vacancy fit estimation.

Answers: "If a university trains specialists with a given curriculum,
what fraction of real market vacancies are they qualified for?"

Uses bootstrapped sampling of real vacancy descriptions to estimate the
distribution of Jaccard fit between curriculum skills and market demands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from hh_competency.analysis.skills import NoDataError
from hh_competency.storage.models import SimulationResult

if TYPE_CHECKING:
    from hh_competency.nlp.pipeline import NLPPipeline
    from hh_competency.storage.db import Database

# Sentinel for filtering vacancies with no extractable skills.
_MIN_VACANCY_SKILLS = 1


class MonteCarloEngine:
    """Monte Carlo simulation for estimating curriculum-vacancy fit distributions.

    Pre-extracts skills from all vacancy descriptions once (expensive NLP pass),
    then performs fast bootstrapped sampling in each iteration.

    Constructor receives externally-managed Database and NLPPipeline —
    pipeline is expensive to initialize (pymorphy3), so it's passed in,
    not created internally.
    """

    def __init__(self, db: Database, pipeline: NLPPipeline) -> None:
        self._db = db
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_curriculum_fit(
        self,
        curriculum_skills: list[str],
        specialty: str,
        n_iter: int = 1000,
        sample_size: int | None = None,
    ) -> SimulationResult:
        """Estimate curriculum-vacancy fit distribution via bootstrapping.

        For each of *n_iter* iterations:
        1.  Sample *sample_size* vacancies (with replacement) from the specialty.
        2.  Compute Jaccard similarity between curriculum skills and each
            sampled vacancy's skills.
        3.  Record the mean Jaccard across sampled vacancies.

        The resulting distribution captures sampling variability — how much
        the fit estimate fluctuates based on which vacancies are sampled.

        Args:
            curriculum_skills: Skill lemmas defining the curriculum.
            specialty: Specialty key (e.g. ``"data_science"``).
            n_iter: Number of bootstrap iterations.
            sample_size: Vacancies to sample per iteration.  Defaults to
                ``min(vacancy_count, 2000)``.

        Returns:
            ``SimulationResult`` with:
            - ``mean_coverage``: mean Jaccard across all iterations.
            - ``ci_lower / ci_upper``: 2.5–97.5% bootstrap confidence interval.
            - ``match_distribution``: per-iteration mean Jaccard values.
            - ``skill_set``: the input curriculum skills.
            - ``n_samples``: *n_iter*.

        Raises:
            NoDataError: If no vacancy descriptions exist for the specialty.
        """
        # --- Pre-extract vacancy skills (single NLP pass) ---
        vacancy_skill_sets, n_total = self._pre_extract_vacancy_skills(specialty)
        if not vacancy_skill_sets:
            raise NoDataError(
                f"No vacancies with extractable skills found for specialty: {specialty}"
            )

        n_available = len(vacancy_skill_sets)
        actual_sample = sample_size or min(n_available, 2000)
        actual_sample = min(actual_sample, n_available)

        curriculum_set = self._normalize_skill_set(curriculum_skills)

        rng = np.random.default_rng()
        fit_values = np.empty(n_iter, dtype=np.float64)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"Simulating curriculum fit for [bold]{specialty}[/] "
                f"({n_iter} iters × {actual_sample} samples)",
                total=n_iter,
            )
            for i in range(n_iter):
                indices = rng.integers(0, n_available, size=actual_sample)
                fit_sum = 0.0
                for idx in indices:
                    vs = vacancy_skill_sets[int(idx)]
                    intersection = len(curriculum_set & vs)
                    union = len(curriculum_set | vs)
                    if union > 0:
                        fit_sum += intersection / union
                fit_values[i] = fit_sum / actual_sample
                progress.update(task, advance=1)

        mean_fit = float(np.mean(fit_values))
        ci_low = float(np.percentile(fit_values, 2.5))
        ci_high = float(np.percentile(fit_values, 97.5))

        return SimulationResult(
            scenario_name="curriculum_fit",
            specialty=specialty,
            n_samples=n_iter,
            mean_coverage=mean_fit,
            ci_lower=ci_low,
            ci_upper=ci_high,
            skill_set=list(curriculum_skills),
            match_distribution=fit_values.tolist(),
        )

    def simulate_skill_impact(
        self,
        curriculum_skills: list[str],
        new_skill: str,
        specialty: str,
        n_iter: int = 500,
    ) -> dict[str, float]:
        """Estimate the impact of adding a single skill to the curriculum.

        Runs a paired comparison: identical vacancy samples for both
        baseline (existing curriculum) and enhanced (+new_skill), so the
        delta reflects the true marginal contribution.

        Args:
            curriculum_skills: Baseline skill lemmas.
            new_skill: Candidate skill lemma to evaluate.
            specialty: Specialty key.
            n_iter: Number of bootstrap iterations.

        Returns:
            Dict with keys:
            - ``baseline_mean``: mean Jaccard without the new skill.
            - ``enhanced_mean``: mean Jaccard with the new skill.
            - ``delta``: enhanced_mean - baseline_mean.
            - ``pct_improvement``: (delta / baseline_mean) * 100.

        Raises:
            NoDataError: If no vacancy descriptions exist for the specialty.
        """
        vacancy_skill_sets, _n_total = self._pre_extract_vacancy_skills(specialty)
        if not vacancy_skill_sets:
            raise NoDataError(
                f"No vacancies with extractable skills found for specialty: {specialty}"
            )

        n_available = len(vacancy_skill_sets)
        sample_size = min(n_available, 2000)

        baseline_set = self._normalize_skill_set(curriculum_skills)
        enhanced_set = baseline_set | {new_skill.lower()}

        rng = np.random.default_rng()
        baseline_values = np.empty(n_iter, dtype=np.float64)
        enhanced_values = np.empty(n_iter, dtype=np.float64)

        # Pre-draw all indices for reproducibility and paired comparison.
        all_indices = rng.integers(0, n_available, size=(n_iter, sample_size))

        for i in range(n_iter):
            indices = all_indices[i]
            b_sum = 0.0
            e_sum = 0.0
            for idx in indices:
                vs = vacancy_skill_sets[int(idx)]
                b_intersect = len(baseline_set & vs)
                e_intersect = len(enhanced_set & vs)
                union = len(enhanced_set | vs)
                if union > 0:
                    b_sum += b_intersect / union
                    e_sum += e_intersect / union
            baseline_values[i] = b_sum / sample_size
            enhanced_values[i] = e_sum / sample_size

        baseline_mean = float(np.mean(baseline_values))
        enhanced_mean = float(np.mean(enhanced_values))
        delta = enhanced_mean - baseline_mean
        pct = (delta / baseline_mean * 100.0) if baseline_mean > 0 else float("inf")

        return {
            "baseline_mean": round(baseline_mean, 6),
            "enhanced_mean": round(enhanced_mean, 6),
            "delta": round(delta, 6),
            "pct_improvement": round(pct, 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pre_extract_vacancy_skills(
        self, specialty: str
    ) -> tuple[list[set[str]], int]:
        """Extract skill lemma-sets from all vacancy descriptions once.

        Returns:
            Tuple of (filtered skill-sets, total vacancy count including
            those with no extractable skills).
        """
        descriptions = self._db.get_vacancy_descriptions(specialty)
        n_total = len(descriptions)

        skill_sets: list[set[str]] = []
        for desc in descriptions:
            keywords = self._pipeline.extract_keywords(desc)
            lemmas = {lemma.lower() for lemma, _pos in keywords}
            if len(lemmas) >= _MIN_VACANCY_SKILLS:
                skill_sets.append(lemmas)

        return skill_sets, n_total

    @staticmethod
    def _normalize_skill_set(skills: list[str]) -> set[str]:
        """Normalize skill lemmas to lowercased set for fast intersection."""
        return {s.lower() for s in skills}
