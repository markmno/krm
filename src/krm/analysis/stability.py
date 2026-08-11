"""Cluster stability testing.

Validates that discovered specialty profiles are statistically stable
via chi-squared differentiation, ANOVA salary analysis, and bootstrap
Rand index for clustering reproducibility.

All methods accept pre-computed data — the module is stateless and has no
database, NLP pipeline, or model dependencies. The caller is responsible
for preparing input arrays and dictionaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2_contingency, f_oneway

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False
    pairwise_tukeyhsd = None  # type: ignore[assignment]


class ClusterStabilityTester:
    """Tests whether competency-role clusters are statistically stable.

    Three complementary tests:

    1. **Chi-squared test of independence**: are skill frequencies
       independent of specialty? (specialty differentiation)
    2. **One-way ANOVA on salary by role**: do role-level mean salaries
       differ significantly?
    3. **Bootstrap adjusted Rand index**: are cluster assignments
       reproducible across random subsamples?

    All methods are stateless — the caller provides pre-computed
    contingency tables, salary groups, or label matrices.
    """

    # ------------------------------------------------------------------
    # Specialty differentiation (chi-squared)
    # ------------------------------------------------------------------

    def test_specialty_differentiation(
        self,
        contingency_table: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Chi-squared test: are skill frequencies independent of specialty?

        Accepts a pre-computed contingency table where each key is a
        specialty name and each value is a list of integer counts (one
        per skill column). Tests the null hypothesis that skill
        frequencies are independent of specialty.

        Args:
            contingency_table: Mapping ``{specialty_name: [count_col0,
                count_col1, ...]}``. All rows must have the same number
                of columns.

        Returns:
            Dict with keys:

            - **chi2_statistic** (*float*): Chi-squared test statistic.
            - **p_value** (*float*): P-value for the independence test.
            - **cramers_v** (*float*): Cramér's V effect size (0–1).
            - **dof** (*int*): Degrees of freedom.
            - **contingency_table** (*list[list[int]]*): Observed
              frequency table (2D).
            - **row_labels** (*list[str]*): Specialty names (rows).
            - **col_labels** (*list[str]*): Zero-based column indices
              as strings.
        """
        row_labels = list(contingency_table.keys())

        if len(row_labels) < 2:
            return _empty_chi2_result()

        rows = list(contingency_table.values())
        n_cols = len(rows[0])

        if n_cols < 2:
            return _empty_chi2_result()
        if any(len(r) != n_cols for r in rows):
            return _empty_chi2_result()

        observed = np.array(rows, dtype=int)

        if observed.sum() < 10:
            return _empty_chi2_result()

        try:
            chi2, p_value, dof, _expected = chi2_contingency(observed)
        except ValueError:
            return _empty_chi2_result()

        # Cramér's V effect size
        n = observed.sum()
        min_dim = min(observed.shape) - 1
        cramers_v = (
            np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
        )

        return {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "cramers_v": round(float(cramers_v), 4),
            "dof": int(dof),
            "contingency_table": observed.tolist(),
            "row_labels": row_labels,
            "col_labels": [str(i) for i in range(n_cols)],
        }

    # ------------------------------------------------------------------
    # Salary by role (ANOVA)
    # ------------------------------------------------------------------

    def test_salary_by_role(
        self,
        role_salaries: dict[str, list[float]],
    ) -> dict[str, Any]:
        """One-way ANOVA: do mean salaries differ significantly by role?

        Accepts pre-computed salary groups. Tests the null hypothesis
        that all role groups have equal mean salary. Optionally runs
        Tukey HSD post-hoc test (requires ``statsmodels``).

        Args:
            role_salaries: Mapping ``{role_name: [salary_value, ...]}``.
                Roles with fewer than 3 observations are excluded.

        Returns:
            Dict with keys:

            - **f_statistic** (*float*): ANOVA F-statistic.
            - **p_value** (*float*): ANOVA p-value.
            - **tukey_hsd** (*list[dict] | None*): Per-pair Tukey HSD
              results. Each dict has keys ``group1``, ``group2``,
              ``meandiff``, ``p_adj``, ``lower``, ``upper``, ``reject``.
              ``None`` if ``statsmodels`` is not installed or the test
              fails.
            - **median_salary_by_role** (*dict[str, float]*): Median
              salary per role.
            - **sample_sizes** (*dict[str, int]*): Number of
              observations per role.
        """
        # Filter roles with at least 3 observations
        valid_roles = {
            role: salaries
            for role, salaries in role_salaries.items()
            if len(salaries) >= 3
        }

        if len(valid_roles) < 2:
            return _empty_anova_result()

        # One-way ANOVA
        salary_groups = list(valid_roles.values())
        try:
            f_stat, p_value = f_oneway(*salary_groups)
        except Exception:
            return _empty_anova_result()

        f_stat = float(f_stat)
        p_value = float(p_value)

        # Tukey HSD (pairwise post-hoc) — optional dependency
        tukey_result: list[dict[str, Any]] | None = None
        if _HAS_STATSMODELS and pairwise_tukeyhsd is not None:
            try:
                all_salaries: list[float] = []
                all_groups: list[str] = []
                for role_name, salaries in valid_roles.items():
                    all_salaries.extend(salaries)
                    all_groups.extend([role_name] * len(salaries))

                tukey = pairwise_tukeyhsd(all_salaries, all_groups, alpha=0.05)

                # Access internal results table
                table_data = getattr(tukey, "_results_table", None)
                body = (
                    table_data.data[1:]  # type: ignore[index]
                    if table_data is not None
                    else []
                )

                tukey_result = []
                for row in body:
                    tukey_result.append({
                        "group1": str(row[0]),
                        "group2": str(row[1]),
                        "meandiff": round(float(row[2]), 2),
                        "p_adj": round(float(row[3]), 4),
                        "lower": round(float(row[4]), 2),
                        "upper": round(float(row[5]), 2),
                        "reject": bool(row[6]),
                    })
            except Exception:
                tukey_result = None

        # Median salary and sample sizes
        median_salary_by_role: dict[str, float] = {}
        sample_sizes: dict[str, int] = {}
        for role_name, salaries in valid_roles.items():
            sample_sizes[role_name] = len(salaries)
            median_salary_by_role[role_name] = round(
                float(np.median(salaries)), 2
            )

        return {
            "f_statistic": round(f_stat, 4),
            "p_value": round(p_value, 6),
            "tukey_hsd": tukey_result,
            "median_salary_by_role": median_salary_by_role,
            "sample_sizes": sample_sizes,
        }

    # ------------------------------------------------------------------
    # Bootstrap Rand index
    # ------------------------------------------------------------------

    def bootstrap_rand_index(
        self,
        labels_matrix: np.ndarray,
        n_iter: int = 50,
        sample_frac: float = 0.8,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Bootstrap adjusted Rand index for clustering stability.

        Accepts a pre-computed label matrix where each column is an
        independent clustering of the same samples. For each bootstrap
        iteration a random subset of rows is drawn with replacement,
        a random pair of clustering columns is selected, and the
        adjusted Rand index (ARI) is computed. High mean ARI with a
        tight confidence interval indicates stable clusterings.

        Args:
            labels_matrix: Integer label array of shape
                ``(n_samples, n_clusterings)``. Each column contains
                cluster assignments for all samples.
            n_iter: Number of bootstrap iterations.
            sample_frac: Fraction of rows to retain per subsample.
            random_state: Seed for the random number generator.

        Returns:
            Dict with keys:

            - **rand_index_mean** (*float*): Mean adjusted Rand index.
            - **rand_index_ci_low** (*float*): 2.5th percentile.
            - **rand_index_ci_high** (*float*): 97.5th percentile.
            - **rand_indices** (*list[float]*): Full list of ARI values.
            - **n_iter** (*int*): Number of successful iterations.
        """
        from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

        labels_matrix = np.asarray(labels_matrix, dtype=int)

        if labels_matrix.ndim != 2:
            return _empty_rand_result()

        n_samples, n_clusterings = labels_matrix.shape

        if n_samples < 10 or n_clusterings < 2:
            return _empty_rand_result()

        rng = np.random.default_rng(random_state)
        n_sample = max(int(n_samples * sample_frac), 10)
        rand_indices: list[float] = []

        for _ in range(n_iter):
            # Subsample rows with replacement
            row_idx = rng.choice(
                n_samples,
                size=min(n_sample, n_samples),
                replace=True,
            )

            # Pick two distinct clustering columns at random
            col_pair = rng.choice(n_clusterings, size=2, replace=False)

            lab_a = labels_matrix[row_idx, col_pair[0]]
            lab_b = labels_matrix[row_idx, col_pair[1]]

            # ARI requires at least 2 unique labels per clustering
            if len(np.unique(lab_a)) < 2 or len(np.unique(lab_b)) < 2:
                continue

            try:
                ari = adjusted_rand_score(lab_a, lab_b)
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


# ======================================================================
# Result helpers
# ======================================================================


def _empty_chi2_result() -> dict[str, Any]:
    """Return a zeroed-out chi-squared result dict."""
    return {
        "chi2_statistic": 0.0,
        "p_value": 1.0,
        "cramers_v": 0.0,
        "dof": 0,
        "contingency_table": [],
        "row_labels": [],
        "col_labels": [],
    }


def _empty_anova_result() -> dict[str, Any]:
    """Return a zeroed-out ANOVA result dict."""
    return {
        "f_statistic": 0.0,
        "p_value": 1.0,
        "tukey_hsd": None,
        "median_salary_by_role": {},
        "sample_sizes": {},
    }


def _empty_rand_result() -> dict[str, Any]:
    """Return a zeroed-out bootstrap Rand index result dict."""
    return {
        "rand_index_mean": 0.0,
        "rand_index_ci_low": 0.0,
        "rand_index_ci_high": 0.0,
        "rand_indices": [],
        "n_iter": 0,
    }
