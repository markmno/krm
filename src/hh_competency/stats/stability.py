"""Cluster stability testing: chi-squared differentiation, ANOVA salary analysis,
and bootstrap Rand index for clustering reproducibility.

Validates that discovered specialty profiles are stable across data samples
and that salary differences between roles are statistically significant.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

from hh_competency.nlp.pipeline import NLPPipeline
from hh_competency.storage.db import Database

# -- Constants --

# Fraction of top skills (per specialty) to include in contingency table
_TOP_SKILLS_FRAC = 0.15
_MIN_TOP_SKILLS = 5
_MAX_TOP_SKILLS = 30


class ClusterStabilityTester:
    """Tests whether competency-role clusters are statistically stable.

    Three complementary tests:
    1. Chi-squared test of independence: are skill frequencies independent
       of specialty? (specialty differentiation)
    2. One-way ANOVA on salary by role: do role-level median salaries differ?
    3. Bootstrap adjusted Rand index: are cluster assignments reproducible
       across random subsamples?
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._pipeline: NLPPipeline | None = None

    def _get_pipeline(self) -> NLPPipeline:
        """Lazy-init NLPPipeline (pymorphy3 is expensive to load)."""
        if self._pipeline is None:
            self._pipeline = NLPPipeline()
        return self._pipeline

    # -- Specialty differentiation (chi-squared) --

    def test_specialty_differentiation(
        self, specialties: list[str]
    ) -> dict:
        """Chi-squared test: are skill frequencies independent of specialty?

        Builds a contingency table: rows = specialties, cols = top skills
        (union of top skills across all specialties). Tests H0: skill
        frequencies are independent of specialty.

        Args:
            specialties: List of specialty keys to compare.

        Returns:
            Dict with keys:
            - chi2_statistic (float): Chi-squared test statistic.
            - p_value (float): P-value for the independence test.
            - cramers_v (float): Cramér's V effect size (0–1).
            - dof (int): Degrees of freedom.
            - contingency_table (list[list[int]]): Observed frequency table.
            - row_labels (list[str]): Specialty names (row labels).
            - col_labels (list[str]): Skill lemma names (column labels).
        """
        if len(specialties) < 2:
            return _empty_chi2_result()

        # Determine column size: reasonable number of top skills to compare
        n_top = self._n_top_skills(specialties)

        # Build contingency table
        all_skills: dict[str, int] = {}
        spec_freqs: dict[str, dict[str, int]] = {}

        for spec in specialties:
            freqs = self._db.get_skill_frequencies(spec, top_n=n_top)
            spec_freqs[spec] = {f.lemma: f.frequency for f in freqs}
            for f in freqs:
                all_skills[f.lemma] = all_skills.get(f.lemma, 0) + 1

        # Select top columns across all specialties
        sorted_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)
        col_lemmas = [lemma for lemma, _freq in sorted_skills[:n_top]]

        # Build 2D table
        table: list[list[int]] = []
        for spec in specialties:
            row = [spec_freqs[spec].get(lemma, 0) for lemma in col_lemmas]
            table.append(row)

        observed = np.array(table, dtype=int)

        # Check minimum expected frequencies for chi-squared validity
        if observed.sum() < 10 or observed.shape[0] < 2 or observed.shape[1] < 2:
            return _empty_chi2_result()

        try:
            chi2, p_value, dof, _expected = chi2_contingency(observed)
        except ValueError:
            return _empty_chi2_result()

        # Cramér's V effect size
        n = observed.sum()
        min_dim = min(observed.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0

        return {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "cramers_v": round(float(cramers_v), 4),
            "dof": int(dof),
            "contingency_table": observed.tolist(),
            "row_labels": list(specialties),
            "col_labels": col_lemmas,
        }

    # -- Salary by role (ANOVA) --

    def test_salary_by_role(self, specialty: str) -> dict:
        """One-way ANOVA: do median salaries differ significantly by role?

        Assigns each vacancy to the best-matching role (by skill overlap
        with defining_skills), then tests H0: all role groups have equal
        mean salary.

        Args:
            specialty: Specialty key.

        Returns:
            Dict with keys:
            - f_statistic (float): ANOVA F-statistic.
            - p_value (float): ANOVA p-value.
            - tukey_hsd (list[dict] | None): Per-pair Tukey HSD results,
              each dict with group1, group2, meandiff, p_adj, lower, upper, reject.
            - median_salary_by_role (dict[str, float]): Median salary per role.
            - sample_sizes (dict[str, int]): Number of vacancies per role.
        """
        # Get descriptions and salary data
        descriptions = self._db.get_vacancy_descriptions(specialty)
        salary_df = self._db.get_salary_data(specialty)

        if descriptions is None or len(descriptions) == 0:
            return _empty_anova_result()
        if salary_df is None or len(salary_df) == 0:
            return _empty_anova_result()

        # Build salary per-description: align descriptions with salary rows
        # get_salary_data and get_vacancy_descriptions both fetch in the same
        # join order (via scrape_runs JOIN raw_vacancies), so they are aligned
        # by index position.
        n_desc = len(descriptions)
        n_salary = len(salary_df)

        # Align: use min length in case of edge mismatch
        n_common = min(n_desc, n_salary)
        descriptions = descriptions[:n_common]
        salary_df = salary_df.iloc[:n_common]

        # Discover roles via co-occurrence clustering
        from hh_competency.analysis.clustering import (  # noqa: PLC0415
            SkillClusterer,
        )

        clusterer = SkillClusterer(self._db)
        roles = clusterer.discover_roles(specialty, n_clusters=3, min_cluster_size=3)

        if not roles:
            return _empty_anova_result()

        # For each vacancy, compute best-matching role by skill overlap
        pipeline = self._get_pipeline()
        role_assignments: list[str | None] = []
        role_salaries: dict[str, list[float]] = defaultdict(list)

        for idx, desc in enumerate(descriptions):
            keywords = pipeline.extract_keywords(desc)
            desc_lemmas = {lemma for lemma, _pos in keywords}

            best_role: str | None = None
            best_overlap = 0

            for role in roles:
                defining = {lemma for lemma, _w in role.defining_skills}
                overlap = len(desc_lemmas & defining)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_role = role.role_name

            role_assignments.append(best_role)

            # Compute salary for this vacancy
            row = salary_df.iloc[idx]
            salary_val = _compute_salary(row)
            if salary_val is not None and best_role is not None:
                role_salaries[best_role].append(salary_val)

        # Filter roles with at least 3 observations for ANOVA
        valid_roles = {r: v for r, v in role_salaries.items() if len(v) >= 3}
        if len(valid_roles) < 2:
            return _empty_anova_result()

        # ANOVA
        salary_groups = list(valid_roles.values())
        try:
            f_stat, p_value = f_oneway(*salary_groups)
        except Exception:
            return _empty_anova_result()

        f_stat = float(f_stat)
        p_value = float(p_value)

        # Tukey HSD (pairwise comparisons)
        tukey_result = None
        try:
            from statsmodels.stats.multicomp import (  # noqa: PLC0415
                pairwise_tukeyhsd,
            )

            all_salaries = []
            all_groups = []
            for role_name, salaries in valid_roles.items():
                all_salaries.extend(salaries)
                all_groups.extend([role_name] * len(salaries))

            tukey = pairwise_tukeyhsd(all_salaries, all_groups, alpha=0.05)
            tukey_df = pd.DataFrame(
                data=tukey._results_table.data[1:],  # noqa: SLF001
                columns=tukey._results_table.data[0],  # noqa: SLF001
            )
            tukey_result = []
            for _, row in tukey_df.iterrows():
                tukey_result.append({
                    "group1": str(row["group1"]),
                    "group2": str(row["group2"]),
                    "meandiff": round(float(row["meandiff"]), 2),
                    "p_adj": round(float(row["p-adj"]), 4),
                    "lower": round(float(row["lower"]), 2),
                    "upper": round(float(row["upper"]), 2),
                    "reject": bool(row["reject"]),
                })
        except Exception:
            tukey_result = None

        # Median salary by role
        median_salary_by_role: dict[str, float] = {}
        sample_sizes: dict[str, int] = {}
        for role_name, salaries in valid_roles.items():
            sample_sizes[role_name] = len(salaries)
            median_salary_by_role[role_name] = round(float(np.median(salaries)), 2)

        return {
            "f_statistic": round(f_stat, 4),
            "p_value": round(p_value, 6),
            "tukey_hsd": tukey_result,
            "median_salary_by_role": median_salary_by_role,
            "sample_sizes": sample_sizes,
        }

    # -- Bootstrap Rand index --

    def bootstrap_rand_index(
        self,
        specialty: str,
        n_iter: int = 50,
        sample_frac: float = 0.8,
        n_clusters: int = 3,
    ) -> dict:
        """Bootstrap adjusted Rand index: how stable are cluster assignments?

        Draws random subsamples of the vacancy set, reclusters skills on each
        subsample, and compares resulting labels to the full-sample clustering
        via adjusted Rand index. High Rand index → stable clustering.

        Args:
            specialty: Specialty key.
            n_iter: Number of bootstrap iterations.
            sample_frac: Fraction of vacancies to retain per subsample.
            n_clusters: Number of clusters for the clustering algorithm.

        Returns:
            Dict with keys:
            - rand_index_mean (float): Mean adjusted Rand index.
            - rand_index_ci_low (float): 2.5th percentile.
            - rand_index_ci_high (float): 97.5th percentile.
            - rand_indices (list[float]): Full list of bootstrap ARI values.
            - n_iter (int): Number of successful iterations.
        """
        from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

        # Build full co-occurrence matrix
        descriptions = self._db.get_vacancy_descriptions(specialty)
        if not descriptions or len(descriptions) < 10:
            return _empty_rand_result()

        skill_list, full_matrix = self._build_cooccurrence_matrix(
            specialty, descriptions
        )
        n_skills = len(skill_list)
        n_vacancies = len(descriptions)

        if n_skills < n_clusters:
            return _empty_rand_result()

        # Full-sample clustering labels
        full_labels = self._cluster_labels(skill_list, full_matrix, n_clusters)
        if full_labels is None:
            return _empty_rand_result()

        # Bootstrap: subsample vacancies → recluster → compare ARI
        rng = np.random.default_rng(42)
        n_sample = max(int(n_vacancies * sample_frac), 10)
        rand_indices: list[float] = []

        for _iter in range(n_iter):
            col_indices = rng.choice(n_vacancies, size=min(n_sample, n_vacancies), replace=True)
            submatrix = full_matrix[:, col_indices]

            # Keep only skills that appear in the subsample
            skill_masks = submatrix.sum(axis=1) > 0
            active_count = int(skill_masks.sum())
            if active_count < n_clusters:
                continue

            # Filter to active skills
            active_indices = np.where(skill_masks)[0]
            active_sub = submatrix[active_indices, :]

            sub_labels = self._cluster_labels(
                [skill_list[i] for i in active_indices],
                active_sub,
                n_clusters,
            )
            if sub_labels is None:
                continue

            # Compare labels only for skills present in both clusterings
            full_sub_labels = full_labels[active_indices]
            try:
                ari = adjusted_rand_score(full_sub_labels, sub_labels)
                rand_indices.append(round(float(ari), 4))
            except Exception:
                continue

        if not rand_indices:
            return _empty_rand_result()

        arr = np.array(rand_indices)
        return {
            "rand_index_mean": round(float(np.mean(arr)), 4),
            "rand_index_ci_low": round(float(np.percentile(arr, 2.5)), 4),
            "rand_index_ci_high": round(float(np.percentile(arr, 97.5)), 4),
            "rand_indices": [round(float(x), 4) for x in rand_indices],
            "n_iter": len(rand_indices),
        }

    # -- Private helpers --

    def _n_top_skills(self, specialties: list[str]) -> int:
        """Compute reasonable column count for contingency table."""
        n_spec = len(specialties)
        if n_spec == 0:
            return _MIN_TOP_SKILLS

        total_skills = 0
        for spec in specialties:
            freqs = self._db.get_skill_frequencies(spec, top_n=200)
            total_skills += len(freqs)

        avg_skills = total_skills // n_spec if n_spec > 0 else 30
        n_top = max(_MIN_TOP_SKILLS, min(int(avg_skills * _TOP_SKILLS_FRAC), _MAX_TOP_SKILLS))
        return n_top

    def _build_cooccurrence_matrix(
        self,
        specialty: str,
        descriptions: list[str],
    ) -> tuple[list[str], np.ndarray]:
        """Build binary skill×vacancy co-occurrence matrix.

        Given pre-fetched descriptions (avoids repeated DB queries).
        """
        freqs = self._db.get_skill_frequencies(specialty, top_n=200)
        skill_list = [f.lemma for f in freqs]

        n_skills = len(skill_list)
        n_vacancies = len(descriptions)
        matrix = np.zeros((n_skills, n_vacancies), dtype=bool)
        skill_to_idx = {s: i for i, s in enumerate(skill_list)}

        pipeline = self._get_pipeline()
        for j, desc in enumerate(descriptions):
            keywords = pipeline.extract_keywords(desc)
            seen: set[str] = set()
            for lemma, _pos in keywords:
                idx = skill_to_idx.get(lemma)
                if idx is not None and lemma not in seen:
                    matrix[idx, j] = True
                    seen.add(lemma)

        return skill_list, matrix

    @staticmethod
    def _cluster_labels(
        skills: list[str],
        matrix: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray | None:
        """Cluster skills and return label array. Returns None if infeasible."""
        from scipy.spatial.distance import pdist, squareform  # noqa: PLC0415
        from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

        n_skills = len(skills)
        if n_skills < n_clusters:
            return None

        n_actual = min(n_clusters, n_skills)
        if n_skills > 1:
            dist_condensed = pdist(matrix, metric="jaccard")
            dist_matrix = squareform(dist_condensed)
        else:
            return np.zeros(1, dtype=int)

        clustering = AgglomerativeClustering(
            n_clusters=n_actual,
            metric="precomputed",
            linkage="average",
        )
        return clustering.fit_predict(dist_matrix)


# -- Result helpers --


def _compute_salary(row: pd.Series) -> float | None:
    """Convert a salary DataFrame row to a single numeric value (RUB).

    Uses mid-point of salary range. Returns None if value is NaN.
    """
    salary_from = row["salary_from"]
    salary_to = row["salary_to"]

    f_val = salary_from if pd.notna(salary_from) else None
    t_val = salary_to if pd.notna(salary_to) else None

    if f_val is not None and t_val is not None:
        return (f_val + t_val) / 2.0
    if f_val is not None:
        return float(f_val)
    if t_val is not None:
        return float(t_val)
    return None


def _empty_chi2_result() -> dict:
    return {
        "chi2_statistic": 0.0,
        "p_value": 1.0,
        "cramers_v": 0.0,
        "dof": 0,
        "contingency_table": [],
        "row_labels": [],
        "col_labels": [],
    }


def _empty_anova_result() -> dict:
    return {
        "f_statistic": 0.0,
        "p_value": 1.0,
        "tukey_hsd": None,
        "median_salary_by_role": {},
        "sample_sizes": {},
    }


def _empty_rand_result() -> dict:
    return {
        "rand_index_mean": 0.0,
        "rand_index_ci_low": 0.0,
        "rand_index_ci_high": 0.0,
        "rand_indices": [],
        "n_iter": 0,
    }
