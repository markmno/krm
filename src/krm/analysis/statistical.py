"""Cluster significance testing via bootstrap silhouette and gap statistic.

Validates whether discovered clusters differ meaningfully from random chance,
and identifies the optimal number of clusters.

Stateless module: all data comes as pre-computed numpy arrays.
No database, NLP pipeline, or file I/O dependencies.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

_DEFAULT_ALPHA = 0.05
_DEFAULT_RANDOM_STATE = 42


class ClusterSignificanceTester:
    """Validates clusters with bootstrap hypothesis tests.

    Uses permutation-based silhouette testing to determine whether
    discovered clusters are significantly different from random labeling,
    and gap statistic to recommend the optimal number of clusters.

    Stateless — call methods directly with pre-computed numpy arrays.
    """

    @staticmethod
    def test_cluster_significance(
        skill_vectors: np.ndarray,
        min_skill_count: int = 3,
        n_iter: int = 100,
        alpha: float = _DEFAULT_ALPHA,
        n_clusters: int = 3,
        random_state: int = _DEFAULT_RANDOM_STATE,
    ) -> dict[str, float | bool | int]:
        """Test whether discovered clusters differ significantly from random.

        Clusters ``skill_vectors`` using AgglomerativeClustering with Jaccard
        distance, computes the real silhouette score, then runs a bootstrap
        permutation test: for each iteration, the real cluster labels are
        randomly permuted and the silhouette score under the null hypothesis
        is recorded. The p-value is the proportion of bootstrap samples
        with silhouette ≥ the observed value.

        Args:
            skill_vectors: 2D numpy array (samples × features).
            min_skill_count: Minimum samples required (fewer → empty result).
            n_iter: Number of bootstrap permutations (50–500 recommended).
            alpha: Significance threshold.
            n_clusters: Number of clusters for the clustering algorithm.
            random_state: Seed for reproducible permutations.

        Returns:
            Dict with keys:
            - silhouette_score (float): Observed silhouette score.
            - silhouette_ci_low (float): 2.5th percentile of bootstrap null.
            - silhouette_ci_high (float): 97.5th percentile of bootstrap null.
            - p_value (float): Proportion of bootstrap samples ≥ real silhouette.
            - is_significant (bool): True if p_value < alpha.
            - n_iter (int): Number of bootstrap iterations performed.
        """
        n_samples = skill_vectors.shape[0]
        if n_samples < min_skill_count or n_samples < 2 * n_clusters:
            return ClusterSignificanceTester._empty_significance_result(n_iter)

        real_labels = ClusterSignificanceTester._cluster_labels(
            skill_vectors, n_clusters
        )
        if real_labels is None:
            return ClusterSignificanceTester._empty_significance_result(n_iter)

        real_silhouette = ClusterSignificanceTester._compute_silhouette(
            skill_vectors, real_labels
        )

        rng = np.random.default_rng(random_state)
        null_silhouettes: list[float] = []

        for _iter in range(n_iter):
            permuted = rng.permutation(real_labels)
            if len(np.unique(permuted)) < 2:
                continue
            try:
                ss = ClusterSignificanceTester._compute_silhouette(
                    skill_vectors, permuted
                )
                null_silhouettes.append(ss)
            except ValueError:
                continue

        if not null_silhouettes:
            return ClusterSignificanceTester._empty_significance_result(n_iter)

        null_arr = np.array(null_silhouettes)
        ci_low = float(np.percentile(null_arr, 2.5))
        ci_high = float(np.percentile(null_arr, 97.5))
        p_value = float(np.mean(null_arr >= real_silhouette))

        return {
            "silhouette_score": round(float(real_silhouette), 4),
            "silhouette_ci_low": round(ci_low, 4),
            "silhouette_ci_high": round(ci_high, 4),
            "p_value": round(p_value, 4),
            "is_significant": bool(p_value < alpha),
            "n_iter": len(null_silhouettes),
        }

    @staticmethod
    def test_cluster_count(
        skill_vectors: np.ndarray,
        max_k: int = 8,
        n_ref: int = 5,
        random_state: int = _DEFAULT_RANDOM_STATE,
    ) -> list[dict[str, int | float | bool | None]]:
        """Evaluate silhouette score and gap statistic for k=2..max_k.

        For each k, clusters the data and computes:
        - Silhouette score (higher = better separation).
        - Gap statistic: log(W_k) - (1/B) * sum(log(W*_k))
          where W_k is within-cluster dispersion and W*_k comes from
          uniform reference datasets.

        Args:
            skill_vectors: 2D numpy array (samples × features).
            max_k: Maximum number of clusters to test.
            n_ref: Number of reference datasets for gap statistic.
            random_state: Seed for reproducible reference data.

        Returns:
            List of dicts (one per k) with keys:
            - k (int): Number of clusters.
            - silhouette_score (float): Observed silhouette at this k.
            - gap_statistic (float or None): Gap statistic value.
            - w_k (float): Within-cluster dispersion (log scale).
            - wk_ref (float): Mean log-dispersion of reference datasets.
            - is_optimal (bool): True if this k has highest silhouette.
        """
        n_samples = skill_vectors.shape[0]
        n_max = min(max_k, n_samples - 1)

        if n_samples < 4:
            return []

        results: list[dict[str, int | float | bool | None]] = []
        rng = np.random.default_rng(random_state)

        best_silhouette = -1.0
        best_k = 2

        for k in range(2, n_max + 1):
            labels = ClusterSignificanceTester._cluster_labels(skill_vectors, k)
            if labels is None:
                continue

            n_unique = len(np.unique(labels))
            if n_unique < 2:
                continue

            ss = ClusterSignificanceTester._compute_silhouette(skill_vectors, labels)
            ss_val = round(float(ss), 4)

            gap_info = ClusterSignificanceTester._gap_statistic(
                skill_vectors, labels, k, n_ref, rng
            )

            result = {
                "k": k,
                "silhouette_score": ss_val,
                "gap_statistic": gap_info["gap"],
                "w_k": gap_info["w_k"],
                "wk_ref": gap_info["wk_ref"],
            }
            results.append(result)

            if ss_val > best_silhouette:
                best_silhouette = ss_val
                best_k = k

        for r in results:
            r["is_optimal"] = bool(r["k"] == best_k)

        return results

    # -- Private helpers --

    @staticmethod
    def _cluster_labels(
        matrix: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray | None:
        """Cluster samples using AgglomerativeClustering with Jaccard distance.

        Args:
            matrix: 2D numpy array (samples × features).
            n_clusters: Desired number of clusters.

        Returns:
            Cluster label array, or None if infeasible.
        """
        n_samples = matrix.shape[0]
        if n_samples < n_clusters:
            return None

        n_actual = min(n_clusters, n_samples)
        if n_samples > 1:
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

    @staticmethod
    def _compute_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score using Jaccard distance (precomputed).

        Matches the distance metric used for AgglomerativeClustering.
        """
        if matrix.shape[0] < 2 or matrix.ndim < 2:
            return 0.0
        try:
            dist_condensed = pdist(matrix, metric="jaccard")
            return float(
                silhouette_score(dist_condensed, labels, metric="precomputed")
            )
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _gap_statistic(
        matrix: np.ndarray,
        labels: np.ndarray,
        k: int,
        n_ref: int,
        rng: np.random.Generator,
    ) -> dict[str, float | None]:
        """Compute gap statistic for the given clustering.

        Gap(k) = (1/B) * Σ log(W*_k) - log(W_k)
        where W_k is the pooled within-cluster sum of pairwise distances.

        Args:
            matrix: 2D numpy array (samples × features).
            labels: Cluster labels for the real data.
            k: Number of clusters.
            n_ref: Number of uniform reference datasets.
            rng: Seeded random generator.

        Returns:
            Dict with keys: gap, w_k, wk_ref.
        """
        w_k = _within_cluster_dispersion(matrix, labels, k)

        log_wk_refs: list[float] = []

        for _ref_iter in range(n_ref):
            ref = rng.uniform(size=matrix.shape)
            ref_labels = AgglomerativeClustering(
                n_clusters=k, linkage="average"
            ).fit_predict(ref)
            w_ref = _within_cluster_dispersion(ref, ref_labels, k)
            if w_ref > 0:
                log_wk_refs.append(np.log(w_ref))

        if not log_wk_refs:
            return {"gap": None, "w_k": 0.0, "wk_ref": 0.0}

        wk_ref = float(np.mean(log_wk_refs))
        log_w_k = np.log(w_k) if w_k > 0 else 0.0
        gap = round(wk_ref - log_w_k, 4)

        return {
            "gap": gap,
            "w_k": round(log_w_k, 4),
            "wk_ref": round(wk_ref, 4),
        }

    @staticmethod
    def _empty_significance_result(n_iter: int) -> dict[str, float | bool | int]:
        """Return a result dict indicating insufficient data for testing."""
        return {
            "silhouette_score": 0.0,
            "silhouette_ci_low": 0.0,
            "silhouette_ci_high": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "n_iter": n_iter,
        }


def _within_cluster_dispersion(
    matrix: np.ndarray, labels: np.ndarray, k: int
) -> float:
    """Compute pooled within-cluster sum of pairwise Euclidean distances.

    For each cluster, computes pairwise distances between all samples
    assigned to that cluster and sums them. Returns the sum across clusters.
    """
    total = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            continue
        sub = matrix[indices]
        dists = pdist(sub, metric="euclidean")
        total += float(np.sum(dists))

    return total
