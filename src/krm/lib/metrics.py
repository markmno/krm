"""Evaluation metrics for KRM pipeline.

Provides sklearn-compatible implementations of F1, Silhouette,
Davies-Bouldin, Adjusted Rand Index, and Spearman rho for
pipeline validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from scipy.stats import spearmanr


def compute_classification_f1(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
) -> dict[str, float | dict[str, float]]:
    """Compute per-class and macro F1 scores."""
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {
            label: f1_score(y_true, y_pred, average=None, labels=labels)[i]
            for i, label in enumerate(labels)
        },
    }


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute Silhouette and Davies-Bouldin for clustering.

    Filters out noise points (label == -1) before computing.
    """
    mask = labels != -1
    if mask.sum() < 2:
        return {"silhouette": 0.0, "davies_bouldin": 0.0, "n_clusters": 0, "n_noise": int((~mask).sum())}

    emb_filt = embeddings[mask]
    lab_filt = labels[mask]
    n_unique = len(set(lab_filt))

    if n_unique < 2:
        return {"silhouette": 0.0, "davies_bouldin": 0.0, "n_clusters": n_unique, "n_noise": int((~mask).sum())}

    return {
        "silhouette": float(silhouette_score(emb_filt, lab_filt)),
        "davies_bouldin": float(davies_bouldin_score(emb_filt, lab_filt)),
        "n_clusters": n_unique,
        "n_noise": int((~mask).sum()),
    }


def compute_bootstrap_ari(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 100,
    sample_fraction: float = 0.8,
    random_seed: int = 42,
) -> dict[str, float]:
    """Bootstrap stability of clustering via Adjusted Rand Index."""
    rng = np.random.default_rng(random_seed)
    mask = labels != -1
    emb_filt = embeddings[mask]
    lab_filt = labels[mask]
    n = len(emb_filt)

    if n < 10:
        return {"ari_mean": 0.0, "ari_std": 0.0, "n_valid_points": n}

    ari_scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=int(n * sample_fraction), replace=False)
        idx_full = np.zeros(n, dtype=bool)
        idx_full[idx] = True

        sub_emb = emb_filt[idx_full]
        sub_lab = lab_filt[idx_full]

        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=min(5, len(sub_emb) // 3))
        sub_pred = clusterer.fit_predict(sub_emb)

        valid = sub_pred != -1
        if valid.sum() < 2:
            continue

        ari = adjusted_rand_score(sub_lab[valid], sub_pred[valid])
        ari_scores.append(ari)

    if not ari_scores:
        return {"ari_mean": 0.0, "ari_std": 0.0, "n_valid_points": n}

    return {
        "ari_mean": float(np.mean(ari_scores)),
        "ari_std": float(np.std(ari_scores)),
        "n_valid_points": n,
    }


def compute_spearman_correlation(
    model_scores: list[list[float]],
    expert_scores: list[list[float]],
) -> dict[str, float]:
    """Spearman correlation between model and expert axis scores.

    Input: two lists of shape (n_roles, n_axes).
    """
    model_flat = [v for row in model_scores for v in row]
    expert_flat = [v for row in expert_scores for v in row]
    rho, p_value = spearmanr(model_flat, expert_flat)
    return {"spearman_rho": float(rho), "p_value": float(p_value)}


def compute_coverage(n_total: int, n_classified: int) -> dict[str, float]:
    """Pipeline coverage: fraction of vacancies fully processed."""
    if n_total == 0:
        return {"coverage": 0.0, "n_total": 0, "n_classified": n_classified}
    return {
        "coverage": n_classified / n_total,
        "n_total": n_total,
        "n_classified": n_classified,
    }
