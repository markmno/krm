"""Cluster significance testing via bootstrap silhouette and gap statistic.

Validates whether discovered competency-role clusters differ meaningfully
from random chance, and identifies the optimal number of clusters.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from hh_competency.nlp.pipeline import NLPPipeline
from hh_competency.storage.db import Database


class ClusterSignificanceTester:
    """Validates competency-role clusters with bootstrap hypothesis tests.

    Uses permutation-based silhouette testing to determine whether
    discovered clusters are significantly different from random labeling,
    and gap statistic to recommend the optimal number of clusters.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._pipeline: NLPPipeline | None = None

    def _get_pipeline(self) -> NLPPipeline:
        """Lazy-init NLPPipeline (pymorphy3 is expensive to load)."""
        if self._pipeline is None:
            self._pipeline = NLPPipeline()
        return self._pipeline

    # -- Public API --

    def test_cluster_significance(
        self,
        specialty: str,
        n_iter: int = 100,
        alpha: float = 0.05,
        n_clusters: int = 3,
    ) -> dict:
        """Test whether discovered clusters differ significantly from random.

        Builds a skill×vacancy co-occurrence matrix, clusters skills using
        AgglomerativeClustering, computes the real silhouette score, then
        runs a bootstrap permutation test: for each iteration, the real
        cluster labels are randomly permuted and the silhouette score under
        the null hypothesis is recorded. The p-value is the proportion of
        bootstrap samples with silhouette ≥ the observed value.

        Args:
            specialty: Specialty key (e.g. 'data_science').
            n_iter: Number of bootstrap permutations (50–500 recommended).
            alpha: Significance threshold (default 0.05).
            n_clusters: Number of clusters for the clustering algorithm.

        Returns:
            Dict with keys:
            - silhouette_score (float): Observed silhouette score.
            - silhouette_ci_low (float): 2.5th percentile of bootstrap null.
            - silhouette_ci_high (float): 97.5th percentile of bootstrap null.
            - p_value (float): Proportion of bootstrap samples ≥ real silhouette.
            - is_significant (bool): True if p_value < alpha.
            - n_iter (int): Number of bootstrap iterations performed.
        """
        skill_list, matrix = self._build_cooccurrence_matrix(specialty)

        if len(skill_list) < 2 * n_clusters:
            return self._empty_significance_result(n_iter)

        # Compute real clusters and observed silhouette
        real_labels = self._cluster_labels(skill_list, matrix, n_clusters)
        if real_labels is None:
            return self._empty_significance_result(n_iter)

        real_silhouette = self._compute_silhouette(matrix, real_labels)

        # Bootstrap: permute labels and recompute silhouette
        rng = np.random.default_rng(42)
        null_silhouettes: list[float] = []

        for _iter in range(n_iter):
            permuted = rng.permutation(real_labels)
            # Skip if all labels are the same after permutation
            if len(np.unique(permuted)) < 2:
                continue
            try:
                ss = self._compute_silhouette(matrix, permuted)
                null_silhouettes.append(ss)
            except ValueError:
                continue

        if not null_silhouettes:
            return self._empty_significance_result(n_iter)

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

    def test_cluster_count(
        self,
        specialty: str,
        max_k: int = 8,
        n_ref: int = 5,
    ) -> list[dict]:
        """Evaluate silhouette score and gap statistic for k=2..max_k.

        For each k, clusters the co-occurrence matrix and computes:
        - Silhouette score (higher = better separation)
        - Gap statistic: log(W_k) - (1/B) * sum(log(W*_k))
          where W_k is within-cluster dispersion and W*_k comes from
          uniform reference datasets.

        Args:
            specialty: Specialty key.
            max_k: Maximum number of clusters to test.
            n_ref: Number of reference datasets for gap statistic.

        Returns:
            List of dicts (one per k) with keys:
            - k (int): Number of clusters.
            - silhouette_score (float): Observed silhouette at this k.
            - gap_statistic (float or None): Gap statistic value.
            - w_k (float): Within-cluster dispersion (log scale).
            - wk_ref (float): Mean log-dispersion of reference datasets.
            - is_optimal (bool): True if this k has highest silhouette.
        """
        skill_list, matrix = self._build_cooccurrence_matrix(specialty)
        n_skills = len(skill_list)
        n_max = min(max_k, n_skills - 1)

        if n_skills < 4:
            return []

        results: list[dict] = []
        rng = np.random.default_rng(42)

        best_silhouette = -1.0
        best_k = 2

        for k in range(2, n_max + 1):
            labels = self._cluster_labels(skill_list, matrix, k)
            if labels is None:
                continue

            n_unique = len(np.unique(labels))
            if n_unique < 2:
                continue

            # Silhouette score
            ss = self._compute_silhouette(matrix, labels)
            ss_val = round(float(ss), 4)

            # Gap statistic
            gap_info = self._gap_statistic(matrix, labels, k, n_ref, rng)

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

        # Mark optimal k
        for r in results:
            r["is_optimal"] = bool(r["k"] == best_k)

        return results

    # -- Private helpers --

    def _build_cooccurrence_matrix(
        self, specialty: str
    ) -> tuple[list[str], np.ndarray]:
        """Build binary skill×vacancy co-occurrence matrix.

        Each row corresponds to a skill lemma, each column to a vacancy.
        Entry [i, j] = 1 if skill i appears in vacancy j.

        Returns:
            Tuple of (skill_list, cooccurrence_matrix).
        """
        descriptions = self._db.get_vacancy_descriptions(specialty)
        if not descriptions:
            return [], np.array([[]], dtype=bool)

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
        """Cluster skills and return label array.

        Returns None if clustering is infeasible (too few skills).
        """
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

    @staticmethod
    def _compute_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score from binary co-occurrence matrix and labels.

        Uses Jaccard distance (which equals 1 - Jaccard similarity) as the metric,
        matching the distance used for AgglomerativeClustering. Since silhouette_score
        requires a feature matrix or precomputed distance, we compute the condensed
        distance matrix and pass it as 'precomputed'.
        """
        if matrix.shape[0] < 2 or matrix.ndim < 2:
            return 0.0
        try:
            dist_condensed = pdist(matrix, metric="jaccard")
            return float(silhouette_score(dist_condensed, labels, metric="precomputed"))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _gap_statistic(
        matrix: np.ndarray,
        labels: np.ndarray,
        k: int,
        n_ref: int,
        rng: np.random.Generator,
    ) -> dict:
        """Compute gap statistic for the given clustering.

        Gap(k) = (1/B) * Σ log(W*_k) - log(W_k)
        where W_k is the pooled within-cluster sum of pairwise distances.

        Returns dict with keys: gap, w_k, wk_ref.
        """
        # Real within-cluster dispersion
        w_k = _within_cluster_dispersion(matrix, labels, k)

        # Null reference: generate B uniform datasets
        log_wk_refs: list[float] = []

        for _ref_iter in range(n_ref):
            # Generate uniform reference data in the same shape
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
    def _empty_significance_result(n_iter: int) -> dict:
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

    For each cluster, computes pairwise distances between all skills
    assigned to that cluster and sums them. Returns the sum across clusters.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)

    total = 0.0
    for _label, indices in groups.items():
        if len(indices) < 2:
            continue
        sub = matrix[indices]
        dists = pdist(sub, metric="euclidean")
        total += float(np.sum(dists))

    return total
