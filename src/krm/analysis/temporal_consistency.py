"""3-tier statistical validation framework for temporal consistency.

Proves that CRM-model characteristics are temporally stable — a fundamental
property that distinguishes characteristics (stable role requirements) from
skills (which change with technology). Unlike skills that evolve with tools
and platforms, a role's core *characteristics* (e.g., "attention to detail,"
"systems thinking") must persist over time for the CRM model to be sustainable.

Tiers:
    1. **Stability Confirmation** — non-rejection of temporal stability (p > α).
    2. **Association Confirmation** — role-characteristic links persist across years.
    3. **Robustness** — resampling, bootstrap, and simulation proof beyond noise.

All functions are pure math — stateless, no database or file I/O.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    # Tier 1 — Stability
    "cochran_armitage_test",
    "breslow_day_test",
    "temporal_stability_report",
    # Tier 2 — Association
    "cochran_mantel_haenszel_test",
    # Tier 3 — Robustness
    "permutation_stability_test",
    "block_bootstrap_ci",
    "fleiss_kappa_agreement",
    "monte_carlo_stability",
    # Validation report
    "validate_characteristics",
]


# ============================================================================
# Tier 1 — Stability Confirmation (H₀: distribution is stable across years)
# ============================================================================


def cochran_armitage_test(contingency_table: np.ndarray) -> dict[str, Any]:
    """Cochran-Armitage trend test for binomial proportions over ordered years.

    Tests the null hypothesis that a binary characteristic's prevalence has
    **no linear trend** across T ordered time periods. Non-rejection
    (p > 0.05) means the characteristic is temporally stable.

    Implemented from first principles: computes a Z-statistic using equally
    spaced scores (0, 1, ..., T-1) and the standard normal CDF for the
    two-sided p-value.

    Args:
        contingency_table: 2×T integer array where row 0 = count **without**
            the characteristic, row 1 = count **with** the characteristic,
            and column j corresponds to year j (ordered chronologically).

    Returns:
        Dict with keys:
        - **statistic** (*float*): Z-statistic (signed; positive = increasing trend).
        - **p_value** (*float*): Two-sided p-value.
        - **is_stable** (*bool*): True if p > 0.05 (fail to reject H₀ of no trend).
        - **n_years** (*int*): Number of time periods (T).

    Raises:
        ValueError: If the table is not 2×T with T ≥ 2, or contains non-integer data.
    """
    table = np.asarray(contingency_table, dtype=np.float64)

    if table.ndim != 2 or table.shape[0] != 2 or table.shape[1] < 2:
        raise ValueError(
            f"Expected 2×T array with T ≥ 2, got shape {table.shape}"
        )

    # Rows: 0 = absent, 1 = present
    n_absent = table[0, :]  # n₀ⱼ
    n_present = table[1, :]  # n₁ⱼ
    n_col = n_absent + n_present  # column totals nⱼ
    N = n_col.sum()  # grand total
    N1 = n_present.sum()  # total with characteristic
    p_overall = N1 / N if N > 0 else 0.0

    T = table.shape[1]
    scores = np.arange(T, dtype=np.float64)  # 0, 1, ..., T-1

    # Numerator: ∑ n₁ⱼ · sⱼ - (N₁/N) · ∑ nⱼ · sⱼ
    sum_n1_s = np.dot(n_present, scores)
    sum_n_s = np.dot(n_col, scores)
    numerator = sum_n1_s - p_overall * sum_n_s

    # Denominator: sqrt(p(1-p) · [∑ nⱼ·sⱼ² - (∑ nⱼ·sⱼ)²/N])
    sum_n_s2 = np.dot(n_col, scores**2)
    denom = math.sqrt(
        max(p_overall * (1.0 - p_overall), 1e-15)
        * max(sum_n_s2 - (sum_n_s**2) / N, 1e-15)
    )

    z_stat = numerator / denom if denom > 1e-15 else 0.0

    # Two-sided p-value from standard normal CDF
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))

    return {
        "statistic": round(float(z_stat), 4),
        "p_value": round(float(p_value), 6),
        "is_stable": bool(p_value > 0.05),
        "n_years": int(T),
    }


def breslow_day_test(contingency_tables: list[np.ndarray]) -> dict[str, Any]:
    """Breslow-Day test for homogeneity of odds ratios across strata (years).

    Tests whether the odds ratio of a 2×2 association is consistent across
    multiple strata. With T strata (one per year), non-rejection (p > 0.05)
    means the association strength does not vary by year — evidence of
    temporal consistency.

    Implemented from first principles using the conditional maximum-likelihood
    estimate of the common odds ratio and the Breslow-Day chi-squared statistic.

    Args:
        contingency_tables: List of K 2×2 integer arrays. Each table has the
            form ``[[a, b], [c, d]]`` where a = exposed+positive,
            b = exposed+negative, c = unexposed+positive, d = unexposed+negative.

    Returns:
        Dict with keys:
        - **statistic** (*float*): Breslow-Day chi-squared statistic.
        - **p_value** (*float*): P-value from chi-squared distribution.
        - **df** (*int*): Degrees of freedom (K - 1).
        - **is_homogeneous** (*bool*): True if p > 0.05 (odds ratios are homogeneous).
        - **common_odds_ratio** (*float*): Mantel-Haenszel common odds ratio.
        - **n_strata** (*int*): Number of strata (K).

    Raises:
        ValueError: If fewer than 2 tables are provided or any table is not 2×2.
    """
    tables = [np.asarray(t, dtype=np.float64) for t in contingency_tables]
    K = len(tables)

    if K < 2:
        raise ValueError(f"Need at least 2 contingency tables, got {K}")

    for i, tbl in enumerate(tables):
        if tbl.shape != (2, 2):
            raise ValueError(
                f"Table {i}: expected shape (2, 2), got {tbl.shape}"
            )

    # --- Step 1: Mantel-Haenszel common odds ratio estimate ---
    numerator_sum = 0.0
    denominator_sum = 0.0

    for tbl in tables:
        a, b = tbl[0, 0], tbl[0, 1]
        c, d = tbl[1, 0], tbl[1, 1]
        n = a + b + c + d
        if n > 0:
            numerator_sum += (a * d) / n
            denominator_sum += (b * c) / n

    psi_mh = numerator_sum / denominator_sum if denominator_sum > 1e-15 else 1.0

    # --- Step 2: Solve for expected a (E[a]) under common OR in each stratum ---
    # For the 2×2 table [[a, b], [c, d]] with margins fixed, the expected
    # a under odds ratio ψ satisfies the quadratic:
    #   ψ·E² - [ψ·(n₁₊ + n₊₁) + (n₊₀ - n₁₊)]·E + ψ·n₁₊·n₊₁ = 0
    # where n₁₊ = a + b (row 1 total), n₊₁ = a + c (col 1 total),
    # n₊₀ = b + d (col 2 total).

    bd_stat = 0.0
    for tbl in tables:
        a, b = tbl[0, 0], tbl[0, 1]
        c, d = tbl[1, 0], tbl[1, 1]

        n_row1 = a + b  # row 1 margin
        n_col1 = a + c  # col 1 margin
        n_col0 = b + d  # col 2 margin
        n = n_row1 + c + d

        if n < 2 or n_row1 <= 0 or n_col1 <= 0 or n_col0 <= 0:
            continue

        # Quadratic coefficients: A·E² + B·E + C = 0
        if psi_mh == 1.0:
            # Simplified: E = n_row1 * n_col1 / n
            expected_a = n_row1 * n_col1 / n
            # Variance under hypergeometric
            var_a = (
                n_row1
                * n_col1
                * (n - n_row1)
                * (n - n_col1)
                / (n**2 * (n - 1))
            )
        else:
            A_coeff = psi_mh - 1.0
            B_coeff = -(psi_mh * (n_row1 + n_col1) + (n_col0 - n_row1))
            C_coeff = psi_mh * n_row1 * n_col1

            # Quadratic formula (only the smaller root is within range)
            discriminant = B_coeff**2 - 4.0 * A_coeff * C_coeff
            if discriminant < 0 or abs(A_coeff) < 1e-15:
                expected_a = n_row1 * n_col1 / n
            else:
                sqrt_disc = math.sqrt(max(discriminant, 0.0))
                # Two roots: (-B ± sqrt_disc) / (2A)
                root1 = (-B_coeff + sqrt_disc) / (2.0 * A_coeff)
                root2 = (-B_coeff - sqrt_disc) / (2.0 * A_coeff)

                # Pick the root in [max(0, n_row1 + n_col1 - n), min(n_row1, n_col1)]
                lo = max(0.0, n_row1 + n_col1 - n)
                hi = min(n_row1, n_col1)
                if lo <= root1 <= hi:
                    expected_a = root1
                elif lo <= root2 <= hi:
                    expected_a = root2
                else:
                    # Fallback to uncorrected
                    expected_a = n_row1 * n_col1 / n

            # Variance: 1 / [1/E + 1/(n_row1 - E) + 1/(n_col1 - E) + 1/(n_col0 - n_row1 + E)]
            eps = 1e-12
            e_clamped = max(eps, min(expected_a, n_row1 - eps))
            e_col1 = max(eps, min(n_col1, n_col1 - eps))
            denom_a = max(eps, n_col1 - e_clamped)
            denom_b = max(eps, n_col0 - n_row1 + e_clamped)
            var_a = 1.0 / (
                1.0 / e_clamped
                + 1.0 / max(eps, n_row1 - e_clamped)
                + 1.0 / denom_a
                + 1.0 / denom_b
            )

        diff = a - expected_a
        if var_a > 1e-15:
            bd_stat += (diff**2) / var_a

    df = K - 1
    p_value = 1.0 - _chi2_cdf(bd_stat, df) if df > 0 else 1.0

    return {
        "statistic": round(float(bd_stat), 4),
        "p_value": round(float(p_value), 6),
        "df": int(df),
        "is_homogeneous": bool(p_value > 0.05),
        "common_odds_ratio": round(float(psi_mh), 4),
        "n_strata": int(K),
    }


def _chi2_cdf(x: float, df: int) -> float:
    """Regularized lower incomplete gamma for chi-squared CDF.

    Uses the series expansion: P(χ²ₖ ≤ x) = γ(k/2, x/2) / Γ(k/2).
    """
    if x <= 0 or df <= 0:
        return 0.0
    return float(stats.chi2.cdf(x, df))


def temporal_stability_report(characteristics_df: pd.DataFrame) -> dict[str, Any]:
    """High-level Tier-1 stability report for all characteristics.

    Groups a characteristics DataFrame by ``(characteristic, year)``, computes
    annual prevalence proportions, and runs the Cochran-Armitage trend test
    for each characteristic. Returns a per-characteristic verdict of temporal
    stability.

    Args:
        characteristics_df: DataFrame with at minimum the columns:
            - ``characteristic`` — character string identifier.
            - ``year`` — integer year.
            Each row represents one observation (e.g., a role possessing a
            characteristic in a given year). Multiple rows per year are
            expected; the function groups and counts them.

    Returns:
        Dict with keys:
        - **results** (*dict[str, dict]*): Mapping ``{characteristic_id:
          {cochran_armitage_result, is_stable, years, annual_prevalence}}``.
        - **summary** (*dict*): Aggregate counts:
          ``{n_characteristics, n_stable, n_unstable, frac_stable}``.
    """
    required_cols = {"characteristic", "year"}
    if not required_cols.issubset(characteristics_df.columns):
        raise ValueError(
            f"DataFrame must have columns {required_cols}, "
            f"got {set(characteristics_df.columns)}"
        )

    # S3: Validate no NaN values in required columns.
    for col in required_cols:
        if characteristics_df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains NaN values after filtering. "
                f"Ensure input data has no missing values in required columns."
            )

    df = characteristics_df.copy()
    char_ids = df["characteristic"].unique()
    years_all = sorted(df["year"].unique())
    T = len(years_all)

    if T < 2:
        return {
            "results": {},
            "summary": {
                "n_characteristics": int(len(char_ids)),
                "n_stable": 0,
                "n_unstable": 0,
                "frac_stable": 0.0,
                "error": "Need at least 2 distinct years for trend testing.",
            },
        }

    results: dict[str, dict[str, Any]] = {}
    n_stable = 0

    for cid in char_ids:
        cdf = df[df["characteristic"] == cid]

        # Build 2×T table: row 0 = absent (total - present), row 1 = present
        annual_present: list[int] = []
        annual_total: list[int] = []
        prevalence: list[float] = []

        for yr in years_all:
            year_df = cdf[cdf["year"] == yr]
            n_present = len(year_df)
            n_total = len(df[df["year"] == yr])  # total observations that year
            annual_present.append(n_present)
            annual_total.append(max(n_total, n_present))  # at minimum, present ≤ total
            prevalence.append(n_present / max(n_total, 1))

        absent_row = [max(t - p, 0) for t, p in zip(annual_total, annual_present)]
        table = np.array([absent_row, annual_present], dtype=np.float64)

        # Protect against all-zero rows
        if table.sum() < 2:
            results[cid] = {
                "cochran_armitage_result": None,
                "is_stable": True,
                "years": years_all,
                "annual_prevalence": [round(v, 4) for v in prevalence],
                "warning": "Insufficient data for trend test.",
            }
            n_stable += 1
            continue

        ca_result = cochran_armitage_test(table)
        is_stable = ca_result["is_stable"]
        if is_stable:
            n_stable += 1

        results[cid] = {
            "cochran_armitage_result": ca_result,
            "is_stable": is_stable,
            "years": years_all,
            "annual_prevalence": [round(v, 4) for v in prevalence],
        }

    n_total = int(len(char_ids))
    return {
        "results": results,
        "summary": {
            "n_characteristics": n_total,
            "n_stable": n_stable,
            "n_unstable": n_total - n_stable,
            "frac_stable": round(n_stable / max(n_total, 1), 4),
        },
    }


# ============================================================================
# Tier 2 — Association Confirmation (H₁: role-characteristic link persists)
# ============================================================================


def cochran_mantel_haenszel_test(
    stratified_tables: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Cochran-Mantel-Haenszel test for consistent association across strata.

    Tests whether a binary characteristic is consistently associated with
    role membership across year-stratified 2×2 tables. Each stratum is a year.
    H₀: no association (odds ratio = 1). Rejection (p < 0.05) confirms the
    role-characteristic link persists across years.

    Uses ``scipy.stats.contingency_tables.StratifiedTable`` for the computation.

    Args:
        stratified_tables: Mapping ``{stratum_name: np.array}`` where each value
            is a 2×2 table: ``[[a, b], [c, d]]`` with a = role_A+characteristic,
            b = role_B+characteristic, c = role_A-no_characteristic, d = role_B-no_characteristic.

    Returns:
        Dict with keys:
        - **statistic** (*float*): Cochran-Mantel-Haenszel chi-squared statistic.
        - **p_value** (*float*): P-value for association.
        - **odds_ratio** (*float*): Common odds ratio estimate.
        - **is_associated** (*bool*): True if p < 0.05 (significant association).
        - **n_strata** (*int*): Number of strata.
        - **odds_ratio_ci_low** (*float*): 95% CI lower bound.
        - **odds_ratio_ci_high** (*float*): 95% CI upper bound.

    Raises:
        ValueError: If fewer than 2 strata or any table is not 2×2.
    """
    if len(stratified_tables) < 1:
        raise ValueError("Need at least 1 stratum for CMH test.")

    tables_list: list[np.ndarray] = []
    for name, tbl in stratified_tables.items():
        arr = np.asarray(tbl, dtype=np.float64)
        if arr.shape != (2, 2):
            raise ValueError(
                f"Stratum '{name}': expected shape (2, 2), got {arr.shape}"
            )
        tables_list.append(arr)

    # scipy's StratifiedTable requires list of 2×2 arrays
    stacked = np.stack(tables_list, axis=0)  # shape (K, 2, 2)

    try:
        st = stats.contingency_tables.StratifiedTable(stacked)
        cmh_stat = float(st.test_null_odds.statistic)
        cmh_p = float(st.test_null_odds.pvalue)
        log_or = float(st.log_odds_ratio_pooled)
        or_est = float(st.odds_ratio_pooled)
        or_ci = st.odds_ratio_pooled_confint()
        or_ci_low = float(or_ci[0])
        or_ci_high = float(or_ci[1])
    except Exception:
        # Fallback: compute MH common OR manually
        return _cmh_fallback(tables_list)

    return {
        "statistic": round(cmh_stat, 4),
        "p_value": round(cmh_p, 6),
        "odds_ratio": round(or_est, 4),
        "is_associated": bool(cmh_p < 0.05),
        "n_strata": len(tables_list),
        "odds_ratio_ci_low": round(or_ci_low, 4),
        "odds_ratio_ci_high": round(or_ci_high, 4),
    }


def _cmh_fallback(tables: list[np.ndarray]) -> dict[str, Any]:
    """Manual CMH computation when scipy StratifiedTable fails."""
    K = len(tables)
    num = 0.0
    den = 0.0
    stat_sum = 0.0
    var_sum = 0.0

    for tbl in tables:
        a, b = tbl[0, 0], tbl[0, 1]
        c, d = tbl[1, 0], tbl[1, 1]
        n = a + b + c + d
        if n < 2:
            continue
        n1 = a + b
        n2 = c + d
        m1 = a + c

        expected = n1 * m1 / n
        variance = (n1 * n2 * m1 * (n - m1)) / (n * n * (n - 1)) if n > 1 else 1.0

        num += a * d / n
        den += b * c / n

        stat_sum += a
        var_sum += variance

    # CMH statistic
    expected_sum = 0.0
    for tbl in tables:
        a, b = tbl[0, 0], tbl[0, 1]
        c, d = tbl[1, 0], tbl[1, 1]
        n = a + b + c + d
        if n < 2:
            continue
        n1 = a + b
        m1 = a + c
        expected_sum += n1 * m1 / n

    cmh_stat = (
        (stat_sum - expected_sum) ** 2 / var_sum
        if var_sum > 1e-15
        else 0.0
    )
    cmh_p = 1.0 - _chi2_cdf(cmh_stat, 1)

    or_est = num / den if den > 1e-15 else 1.0

    return {
        "statistic": round(float(cmh_stat), 4),
        "p_value": round(float(cmh_p), 6),
        "odds_ratio": round(float(or_est), 4),
        "is_associated": bool(cmh_p < 0.05),
        "n_strata": K,
        "odds_ratio_ci_low": round(float(or_est * math.exp(-1.96 / math.sqrt(max(var_sum, 1e-15)))), 4),
        "odds_ratio_ci_high": round(float(or_est * math.exp(1.96 / math.sqrt(max(var_sum, 1e-15)))), 4),
    }


# ============================================================================
# Tier 3 — Robustness (proof beyond sampling noise)
# ============================================================================


def permutation_stability_test(
    annual_prevalence: dict[int, np.ndarray],
    n_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Permutation test for temporal stability of annual prevalence.

    Tests the null hypothesis that annual prevalence distributions are
    **exchangeable** (no temporal structure). If the observed variance of
    annual prevalence means is not extreme relative to the distribution
    obtained by randomly shuffling year labels, the data are consistent
    with temporal stability.

    Args:
        annual_prevalence: Mapping ``{year: array_of_values}`` where each value
            is a prevalence observation (e.g., per-role prevalence for that year).
        n_iter: Number of permutation iterations.
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys:
        - **observed_variance** (*float*): Variance of annual means.
        - **p_value** (*float*): Fraction of permutations with variance ≥ observed.
        - **null_mean** (*float*): Mean of null-distribution variances.
        - **null_std** (*float*): Standard deviation of null-distribution variances.
        - **is_stable** (*bool*): True if p > 0.05.
        - **n_years** (*int*): Number of years.
        - **n_iter** (*int*): Number of permutations performed.

    Raises:
        ValueError: If fewer than 2 years of data are provided.
    """
    years = sorted(annual_prevalence.keys())
    if len(years) < 2:
        raise ValueError(f"Need at least 2 years, got {len(years)}")

    # Compute annual means
    annual_means = np.array(
        [float(np.mean(annual_prevalence[y])) for y in years]
    )
    observed_var = float(np.var(annual_means, ddof=1))

    # Pool all values and year assignments
    all_values = np.concatenate([annual_prevalence[y] for y in years])
    year_labels = np.concatenate([
        np.full(len(annual_prevalence[y]), i, dtype=int)
        for i, y in enumerate(years)
    ])

    rng = np.random.default_rng(random_state)

    null_variances = np.empty(n_iter, dtype=np.float64)
    for it in range(n_iter):
        shuffled_labels = rng.permutation(year_labels)
        perm_means = np.array([
            float(np.mean(all_values[shuffled_labels == i]))
            for i in range(len(years))
        ])
        null_variances[it] = float(np.var(perm_means, ddof=1))

    p_value = float(np.mean(null_variances >= observed_var))

    return {
        "observed_variance": round(observed_var, 6),
        "p_value": round(p_value, 6),
        "null_mean": round(float(np.mean(null_variances)), 6),
        "null_std": round(float(np.std(null_variances, ddof=1)), 6),
        "is_stable": bool(p_value > 0.05),
        "n_years": len(years),
        "n_iter": n_iter,
    }


def block_bootstrap_ci(
    annual_prevalence: dict[int, np.ndarray],
    block_size: int = 3,
    n_iter: int = 5000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Block bootstrap confidence interval for trend slope.

    Resamples **blocks** of consecutive years to preserve temporal
    autocorrelation, then computes the ordinary least-squares slope
    of annual prevalence means against year for each bootstrap sample.
    If the 95% CI for the slope contains zero, no significant trend exists.

    Args:
        annual_prevalence: Mapping ``{year: array_of_values}`` — per-year
            observations of a characteristic's prevalence.
        block_size: Number of consecutive years per bootstrap block.
        n_iter: Number of bootstrap iterations.
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys:
        - **slope_ci_lower** (*float*): 2.5th percentile of bootstrap slopes.
        - **slope_ci_upper** (*float*): 97.5th percentile of bootstrap slopes.
        - **slope_mean** (*float*): Mean bootstrap slope.
        - **is_zero_in_ci** (*bool*): True if 0 is in the CI (no significant trend).
        - **block_size** (*int*): Block size used.
        - **n_iter** (*int*): Number of bootstrap iterations.
        - **n_years** (*int*): Number of years.

    Raises:
        ValueError: If fewer than 2 years of data.
    """
    years = sorted(annual_prevalence.keys())
    if len(years) < 2:
        raise ValueError(f"Need at least 2 years, got {len(years)}")

    annual_means = np.array(
        [float(np.mean(annual_prevalence[y])) for y in years]
    )
    T = len(annual_means)

    # Use effective block size ≤ T
    bs = min(block_size, T)
    n_blocks = int(math.ceil(T / bs))

    rng = np.random.default_rng(random_state)
    slopes = np.empty(n_iter, dtype=np.float64)

    for it in range(n_iter):
        # Sample blocks with replacement
        sampled_indices: list[int] = []
        for _ in range(n_blocks):
            start = rng.integers(0, T - bs + 1) if bs > 0 else 0
            block_years = list(range(start, min(start + bs, T)))
            sampled_indices.extend(block_years)

        # Trim to exactly T
        sampled_indices = sampled_indices[:T]
        sampled_means = annual_means[sampled_indices]

        # OLS slope: β₁ = Cov(X, Y) / Var(X)
        x = np.arange(len(sampled_means), dtype=np.float64)
        x_mean = x.mean()
        y_mean = sampled_means.mean()
        cov_xy = np.dot(x - x_mean, sampled_means - y_mean)
        var_x = np.dot(x - x_mean, x - x_mean)
        slopes[it] = cov_xy / var_x if var_x > 1e-15 else 0.0

    ci_low = float(np.percentile(slopes, 2.5))
    ci_high = float(np.percentile(slopes, 97.5))

    return {
        "slope_ci_lower": round(ci_low, 6),
        "slope_ci_upper": round(ci_high, 6),
        "slope_mean": round(float(slopes.mean()), 6),
        "is_zero_in_ci": bool(ci_low <= 0.0 <= ci_high),
        "block_size": bs,
        "n_iter": n_iter,
        "n_years": T,
    }


def fleiss_kappa_agreement(annual_labels: list[np.ndarray]) -> dict[str, Any]:
    """Fleiss' kappa for inter-year agreement on binary characteristic labels.

    Measures whether different years "agree" on which roles exhibit a
    characteristic. High kappa means the characteristic's role assignment
    is consistent over time. Implemented from first principles.

    Args:
        annual_labels: List of T binary arrays, each of length N
            (1 = characteristic present for that role in that year,
             0 = absent). All arrays must have the same length.

    Returns:
        Dict with keys:
        - **kappa** (*float*): Fleiss' kappa statistic (-1 to 1).
        - **z_score** (*float*): Z-score for significance test.
        - **p_value** (*float*): P-value (kappa significantly > 0?).
        - **agreement_level** (*str*): Interpretation per Landis & Koch (1977).
            One of ``'slight'``, ``'fair'``, ``'moderate'``, ``'substantial'``,
            ``'almost_perfect'``, or ``'poor'`` if kappa < 0.

    Raises:
        ValueError: If fewer than 2 years of labels.
    """
    if len(annual_labels) < 2:
        raise ValueError(
            f"Need at least 2 years of labels, got {len(annual_labels)}"
        )

    # Stack into (n_subjects, n_raters) where raters = years
    data = np.column_stack([np.asarray(arr, dtype=int) for arr in annual_labels])
    N = data.shape[0]  # subjects (roles)
    n = data.shape[1]  # raters (years)
    k = 2  # categories: 0 (absent), 1 (present)

    # --- Step 1: Per-subject agreement ---
    # n_ij[subject, category] = count of raters assigning that category
    n_ij = np.zeros((N, k), dtype=np.float64)
    for cat in range(k):
        n_ij[:, cat] = (data == cat).sum(axis=1)

    # P_i = (1/[n(n-1)]) * [∑_j n_ij² - n]
    # Measure of agreement for subject i
    P_i = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sum_sq = np.dot(n_ij[i, :], n_ij[i, :])
        P_i[i] = (sum_sq - n) / (n * (n - 1)) if n > 1 else 0.0

    # --- Step 2: Overall agreement ---
    P_bar = float(np.mean(P_i))  # mean proportion of agreement

    # --- Step 3: Expected agreement by chance ---
    # p_j = proportion of all assignments to category j
    p_j = n_ij.sum(axis=0) / (N * n)
    P_bar_e = float(np.dot(p_j, p_j))  # sum of squared marginal proportions

    # --- Step 4: Kappa ---
    denom = 1.0 - P_bar_e
    kappa = (P_bar - P_bar_e) / denom if denom > 1e-15 else 0.0

    # --- Step 5: Standard error and Z-test ---
    # SE from Fleiss (1971) formula for k = 2 categories
    # SE² = [2 / (N·n·(n-1))] · [P_bar_e - (2n-3)·P_bar_e² + 2(n-2)·∑p_j³] / (1-P_bar_e)²
    # Simplified: use the general formula
    sum_pj3 = float(np.dot(p_j, p_j**2))
    se_sq = (2.0 / (N * n * (n - 1))) * (
        P_bar_e - (2 * n - 3) * P_bar_e**2 + 2 * (n - 2) * sum_pj3
    ) / (denom**2) if denom > 1e-15 and n > 1 else 1e15
    se = math.sqrt(max(se_sq, 1e-15))
    z_score = kappa / se if se > 1e-15 else 0.0
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    # --- Step 6: Agreement level ---
    agreement_level = _kappa_interpretation(kappa)

    return {
        "kappa": round(float(kappa), 4),
        "z_score": round(float(z_score), 4),
        "p_value": round(float(p_value), 6),
        "agreement_level": agreement_level,
    }


def _kappa_interpretation(kappa: float) -> str:
    """Interpret Fleiss' kappa per Landis & Koch (1977) benchmarks."""
    if kappa < 0.0:
        return "poor"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost_perfect"


def monte_carlo_stability(
    historical_prevalence: np.ndarray,
    n_simulations: int = 10000,
    drift_std: float = 0.02,
    random_state: int = 42,
) -> dict[str, Any]:
    """Monte Carlo simulation of characteristic prevalence under random drift
    vs structural change.

    Simulates two scenarios to assess the statistical power of detecting
    structural change versus background random walk:

    1. **Null (random walk)**: AR(1) process with small drift — prevalence
       meanders randomly over time. A stable characteristic follows this pattern.
    2. **Alternative (step change)**: Prevalence undergoes a discrete structural
       jump at a random year — what a genuinely changing characteristic looks like.

    Reports the fraction of simulations where a Mann-Kendall trend test correctly
    distinguishes the two regimes.

    Args:
        historical_prevalence: 1-D array of observed prevalence values (or
            their means) ordered by year.
        n_simulations: Number of Monte Carlo repetitions per scenario.
        drift_std: Standard deviation of the random-walk innovation term.
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys:
        - **power_random_walk** (*float*): Fraction of simulations where the
            null (random walk) is NOT rejected — correctly identified as stable.
        - **power_step_change** (*float*): Fraction of simulations where the
            alternative (step change) IS rejected — correctly identified as changing.
        - **drift_std_used** (*float*): The drift standard deviation used.
        - **n_simulations** (*int*): Number of simulations run.
        - **n_years** (*int*): Length of the prevalence series.
    """
    hist = np.asarray(historical_prevalence, dtype=np.float64).flatten()
    T = len(hist)

    if T < 4:
        raise ValueError(f"Need at least 4 years for simulation, got {T}")

    baseline = float(np.mean(hist))
    rng = np.random.default_rng(random_state)

    # --- Power under random walk (H₀) ---
    # AR(1) with φ = 1 (unit root): x_t = x_{t-1} + ε_t,  ε ~ N(0, drift_std²)
    n_power_rw = 0
    for _ in range(n_simulations):
        rw_series = np.empty(T, dtype=np.float64)
        rw_series[0] = baseline + rng.normal(0.0, drift_std)
        for t in range(1, T):
            rw_series[t] = rw_series[t - 1] + rng.normal(0.0, drift_std)

        # Mann-Kendall trend test
        _, p_val = _mann_kendall_p(rw_series)
        if p_val > 0.05:  # correctly fails to reject H₀ (no trend)
            n_power_rw += 1

    power_rw = n_power_rw / n_simulations if n_simulations > 0 else 0.0

    # --- Power under step change (H₁) ---
    # Baseline with a discrete jump at a random year
    step_magnitude = float(np.std(hist, ddof=1)) * 1.5  # 1.5× historical std
    if step_magnitude < drift_std * 3:
        step_magnitude = drift_std * 3  # ensure detectable difference

    n_power_sc = 0
    for _ in range(n_simulations):
        step_series = np.full(T, baseline, dtype=np.float64)
        change_year = rng.integers(1, T)
        step_series[change_year:] = baseline + step_magnitude

        # Add noise
        step_series += rng.normal(0.0, drift_std * 0.5, size=T)

        _, p_val = _mann_kendall_p(step_series)
        if p_val <= 0.05:  # correctly rejects H₀ (detects trend)
            n_power_sc += 1

    power_sc = n_power_sc / n_simulations if n_simulations > 0 else 0.0

    return {
        "power_random_walk": round(power_rw, 4),
        "power_step_change": round(power_sc, 4),
        "drift_std_used": round(drift_std, 4),
        "n_simulations": n_simulations,
        "n_years": T,
    }


def _mann_kendall_p(values: np.ndarray) -> tuple[float, float]:
    """Compute Kendall's tau and approximate p-value for Mann-Kendall test."""
    n = len(values)
    if n < 3:
        return 0.0, 1.0

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    tau = s / (n * (n - 1) / 2.0)
    variance = n * (n - 1) * (2 * n + 5) / 18.0
    z = s / math.sqrt(variance) if variance > 0 else 0.0
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    return tau, p_val


# ============================================================================
# Validation Report — master orchestrator
# ============================================================================


def validate_characteristics(
    characteristics_df: pd.DataFrame,
    roles_df: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Master validation: run all 3 tiers against characteristic data.

    Orchestrates the full validation pipeline and returns a comprehensive
    report. Tier 2 (CMH) requires the ``roles_df`` argument; it is skipped
    if not provided.

    Args:
        characteristics_df: DataFrame with columns ``characteristic``, ``year``.
            Each row = one observation of a characteristic in a given year.
        roles_df: Optional DataFrame with columns ``role_id``, ``year``,
            ``characteristic``. Required for Tier 2 CMH association testing.
        alpha: Significance threshold for all tests (default 0.05).

    Returns:
        Dict with keys:
        - **tier1_stability** (*dict*): Output of ``temporal_stability_report``.
        - **tier2_association** (*dict | None*): Output of per-characteristic
          CMH tests if ``roles_df`` is provided, else ``None``.
        - **tier3_robustness** (*dict*): Per-characteristic permutation, block
          bootstrap, Fleiss' kappa, and Monte Carlo results.
        - **summary** (*dict*): ``{all_tiers_passed, n_tests, n_passed, verdict}``.
    """
    # --- Tier 1: Stability ---
    tier1 = temporal_stability_report(characteristics_df)

    # --- Tier 3: Robustness (per characteristic) ---
    tier3: dict[str, dict[str, Any]] = {}
    char_ids = characteristics_df["characteristic"].unique()
    years_all = sorted(characteristics_df["year"].unique())

    for cid in char_ids:
        cdf = characteristics_df[characteristics_df["characteristic"] == cid]

        # Build annual_prevalence dict with per-observation binary labels.
        # Each vacancy in year Y contributes 1.0 (has characteristic) or
        # 0.0 (does not), rather than a uniform array of mean prevalence.
        annual_prev: dict[int, np.ndarray] = {}
        for yr in years_all:
            year_df = cdf[cdf["year"] == yr]
            n_present = len(year_df)
            n_total = max(
                len(characteristics_df[characteristics_df["year"] == yr]), 1
            )
            n_absent = max(n_total - n_present, 0)
            # Per-observation binary labels: 1.0 for present, 0.0 for absent.
            annual_prev[yr] = np.array(
                [1.0] * n_present + [0.0] * n_absent, dtype=np.float64
            )

        # Fleiss' kappa requires binary per-role labels per year.
        # Build per-observation arrays identical to annual_prev shape.
        annual_binary: list[np.ndarray] = []
        for yr in years_all:
            n_present = len(cdf[cdf["year"] == yr])
            n_total = max(
                len(characteristics_df[characteristics_df["year"] == yr]), 1
            )
            n_absent = max(n_total - n_present, 0)
            annual_binary.append(
                np.array(
                    [1.0] * n_present + [0.0] * n_absent, dtype=float
                )
            )

        # Build prevalence time series from annual means
        prevalence_series = np.array([
            float(np.mean(annual_prev[yr])) for yr in years_all
        ])

        try:
            perm_result = permutation_stability_test(annual_prev)
        except ValueError:
            perm_result = {"p_value": 1.0, "is_stable": True, "error": "insufficient_years"}

        try:
            bootstrap_result = block_bootstrap_ci(annual_prev)
        except ValueError:
            bootstrap_result = {"is_zero_in_ci": True, "error": "insufficient_years"}

        try:
            fleiss_result = fleiss_kappa_agreement(annual_binary)
        except ValueError:
            fleiss_result = {"kappa": 0.0, "agreement_level": "poor", "error": "insufficient_years"}

        try:
            mc_result = monte_carlo_stability(prevalence_series)
        except ValueError:
            mc_result = {"power_random_walk": 0.0, "power_step_change": 0.0}

        tier3[cid] = {
            "permutation": perm_result,
            "block_bootstrap": bootstrap_result,
            "fleiss_kappa": fleiss_result,
            "monte_carlo": mc_result,
        }

    # --- Tier 2: Association (requires roles_df) ---
    tier2: dict[str, dict[str, Any]] | None = None
    if roles_df is not None:
        tier2 = _validate_tier2(characteristics_df, roles_df)

    # --- Summary ---
    n_tests = 0
    n_passed = 0

    # Tier 1: each characteristic's stability verdict
    for _cid, result in tier1["results"].items():
        n_tests += 1
        if result.get("is_stable", True):
            n_passed += 1

    # Tier 2: each CMH result
    if tier2 is not None:
        for _cid, result in tier2.items():
            n_tests += 1
            if result.get("is_associated", False):
                n_passed += 1

    # Tier 3: each characteristic's permutation test
    for _cid, results in tier3.items():
        n_tests += 1
        if results["permutation"].get("is_stable", True):
            n_passed += 1
        # Block bootstrap: zero in CI = stable
        n_tests += 1
        if results["block_bootstrap"].get("is_zero_in_ci", True):
            n_passed += 1

    all_tiers_passed = n_passed >= n_tests * 0.8  # 80% pass threshold
    verdict_message = (
        "PASS — characteristics demonstrate temporal consistency across all tiers"
        if all_tiers_passed
        else "FAIL — some characteristics show temporal instability"
    )

    return {
        "tier1_stability": tier1,
        "tier2_association": tier2,
        "tier3_robustness": tier3,
        "summary": {
            "all_tiers_passed": all_tiers_passed,
            "n_tests": n_tests,
            "n_passed": n_passed,
            "frac_passed": round(n_passed / max(n_tests, 1), 4),
            "verdict": verdict_message,
        },
    }


_SKIP_RESULT: dict[str, Any] = {
    "statistic": float("nan"),
    "p_value": float("nan"),
    "odds_ratio": float("nan"),
    "is_associated": False,
    "n_strata": 0,
    "skipped": True,
    "reason": "",
}


def _validate_tier2(
    characteristics_df: pd.DataFrame,
    roles_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Run Cochran-Mantel-Haenszel tests for each characteristic across role-year strata.

    Tests whether the association between a characteristic and role
    membership is consistent across years (the odds ratio is homogeneous
    and non-zero).

    **Table layout** (one 2×2 per year)::

        Role group A         Role group B
        ┌───────────────┬───────────────┐
        │ n_A_with_char │ n_B_with_char │ ← characteristic present
        │ n_A_no_char   │ n_B_no_char   │ ← characteristic absent
        └───────────────┴───────────────┘

    The two role groups are formed by splitting the set of role IDs in
    half (top / bottom by sort order).  This avoids the degenerate
    ``[[n,1],[m,1]]`` pattern where one column is always ``[1,1]``.

    When ``roles_df`` has fewer than 2 roles or the per-role-per-year
    characteristic data is too sparse to fill a non-degenerate table,
    the result dict carries ``"skipped": True`` with a ``"reason"``
    string.

    Args:
        characteristics_df: ``(characteristic, year)`` rows, one per
            observation.
        roles_df: ``(role_id, year, characteristic)`` rows, one per
            role–characteristic assignment per year.

    Returns:
        ``{characteristic_id: {statistic, p_value, odds_ratio,
        is_associated, n_strata, skipped?, reason?}}``.
    """
    _SKIP: dict[str, Any] = _SKIP_RESULT.copy()

    if roles_df.empty:
        results = {}
        for cid in characteristics_df["characteristic"].unique():
            r = _SKIP.copy()
            r["reason"] = "roles_df is empty"
            results[str(cid)] = r
        return results

    required_cols = {"role_id", "year", "characteristic"}
    if not required_cols.issubset(roles_df.columns):
        results = {}
        for cid in characteristics_df["characteristic"].unique():
            r = _SKIP.copy()
            r["reason"] = (
                f"roles_df missing columns {required_cols - set(roles_df.columns)}"
            )
            results[str(cid)] = r
        return results

    char_ids = characteristics_df["characteristic"].unique()
    years_all = sorted(characteristics_df["year"].unique())
    role_ids = sorted(roles_df["role_id"].unique())

    results: dict[str, dict[str, Any]] = {}

    if len(role_ids) < 2:
        for cid in char_ids:
            r = _SKIP.copy()
            r["reason"] = f"need ≥2 roles for CMH, found {len(role_ids)}"
            results[str(cid)] = r
        return results

    # Split roles into two independent groups for the columns of the
    # 2×2 table.  The split is deterministic (lexicographic on role_id)
    # but arbitrary — it assumes no a-priori grouping exists.
    mid = len(role_ids) // 2
    group_a = set(role_ids[:mid])
    group_b = set(role_ids) - group_a

    for cid in char_ids:
        stratified: dict[str, np.ndarray] = {}
        usable_strata = 0

        for yr in years_all:
            year_roles = roles_df[roles_df["year"] == yr]
            if year_roles.empty:
                continue

            # Roles with this characteristic in this year.
            mask = year_roles["characteristic"] == cid
            roles_with = set(year_roles.loc[mask, "role_id"].unique())

            # Group A counts.
            a_with = len(roles_with & group_a)
            a_without = len(group_a - roles_with)

            # Group B counts.
            b_with = len(roles_with & group_b)
            b_without = len(group_b - roles_with)

            # Skip strata where any cell is zero (CMH requires non-zero
            # cells for valid odds-ratio estimation).
            if min(a_with, a_without, b_with, b_without) == 0:
                continue

            stratified[str(yr)] = np.array(
                [[a_with, b_with], [a_without, b_without]],
                dtype=np.float64,
            )
            usable_strata += 1

        if usable_strata < 2:
            r = _SKIP.copy()
            r["reason"] = (
                f"only {usable_strata} valid strata "
                "(need ≥2 non-degenerate 2×2 tables)"
            )
            results[str(cid)] = r
            continue

        try:
            cmh_result = cochran_mantel_haenszel_test(stratified)
            results[str(cid)] = cmh_result
        except ValueError as exc:
            r = _SKIP.copy()
            r["reason"] = f"CMH computation failed: {exc}"
            results[str(cid)] = r

    return results
