"""Semantic embedding module using navec for Russian text.

Provides skill vectorization, cosine similarity, and semantic clustering
for competency characteristic discovery.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

_NAVEC_MODEL: str = "navec_news_v1_1B_250K_300d_100q.tar"
_NAVEC_URL: str = f"https://storage.yandexcloud.net/natasha-navec/packs/{_NAVEC_MODEL}"
_NAVEC_HOME: Path = Path.home() / ".cache" / "navec"

_VECTOR_DIM: int = 300


class SkillEmbedder:
    """Wraps navec for Russian skill-to-vector conversion."""

    def __init__(self) -> None:
        self._model = self._load_navec()

    def _load_navec(self):
        """Load navec news model, downloading if needed."""
        from navec import Navec

        path = _NAVEC_HOME / _NAVEC_MODEL
        if not path.exists():
            logger.info("Downloading navec model (25 MB)...")
            path = _ensure_navec_downloaded()
        return Navec.load(path)

    def embed(self, skill: str) -> np.ndarray:
        """Convert a skill lemma to 300-dimensional vector.

        For multi-word skills (e.g. "машинный обучение"),
        averages the embeddings of individual words.
        Returns zero vector if no tokens are in vocabulary.
        """
        tokens = skill.lower().split()
        vectors = []
        for token in tokens:
            if token in self._model:
                vectors.append(self._model[token])
        if not vectors:
            return np.zeros(_VECTOR_DIM, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def embed_batch(self, skills: list[str]) -> np.ndarray:
        """Convert a list of skills to a (n, 300) matrix."""
        return np.array([self.embed(s) for s in skills], dtype=np.float32)

    def similarity(self, skill_a: str, skill_b: str) -> float:
        """Cosine similarity between two skills."""
        va = self.embed(skill_a)
        vb = self.embed(skill_b)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def cluster_by_semantics(
        self,
        skills: list[str],
        max_clusters: int = 9,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.4,
    ) -> list[tuple[str, list[str], float]]:
        """Cluster skills by semantic similarity.

        Uses agglomerative clustering with cosine distance on navec vectors.

        Returns:
            List of (cluster_label, member_skills, mean_intra_similarity)
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances

        if len(skills) < min_cluster_size:
            return []

        vectors = self.embed_batch(skills)

        # Filter zero vectors (skills not in navec vocab)
        norms = np.linalg.norm(vectors, axis=1)
        mask = norms > 1e-8
        if mask.sum() < min_cluster_size:
            return []

        valid_skills = [s for s, m in zip(skills, mask, strict=False) if m]
        valid_vecs = vectors[mask]

        n_samples = len(valid_skills)
        if n_samples < min_cluster_size:
            return []

        # Compute cosine distance matrix
        dist_matrix = cosine_distances(valid_vecs)

        # Try different k values, pick best by silhouette
        best_k = min(max_clusters, max(2, n_samples // min_cluster_size))
        k_candidates = list(range(2, best_k + 1))

        if not k_candidates:
            return []

        from sklearn.metrics import silhouette_score

        best_score = -1.0
        best_labels: np.ndarray | None = None

        for k in k_candidates:
            if n_samples < k:
                continue
            clustering = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
            labels = clustering.fit_predict(dist_matrix)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(dist_matrix, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            return []

        # Build clusters
        clusters: dict[int, list[str]] = {}
        for skill, label in zip(valid_skills, best_labels, strict=False):
            clusters.setdefault(int(label), []).append(skill)

        # Filter small clusters, name them, compute intra-similarity
        result: list[tuple[str, list[str], float]] = []
        for _label, members in sorted(clusters.items()):
            if len(members) < min_cluster_size:
                continue
            # Name the cluster: pick the most frequent skill as label
            label_skill = self._name_cluster(members)
            # Mean pairwise similarity within cluster
            indices = [i for i, s in enumerate(valid_skills) if s in members]
            if len(indices) >= 2 and len(indices) <= dist_matrix.shape[0]:
                intra_sim = 1.0 - float(dist_matrix[np.ix_(indices, indices)].mean())
            else:
                intra_sim = 1.0
            result.append((label_skill, members, intra_sim))

        return result

    def _name_cluster(self, skills: list[str], max_name_len: int = 60) -> str:
        """Generate a human-readable name for a skill cluster.

        Picks the top skill (most specific/longest) and appends related
        skills as a description.
        """
        if not skills:
            return "Неизвестный кластер"
        # Prioritize multi-word skills (they're more descriptive)
        multi_word = [s for s in skills if " " in s]
        candidates = multi_word if multi_word else skills
        primary = max(candidates, key=len)
        # Add secondary skills to name
        others = [s for s in skills if s != primary][:2]
        if others:
            suffix = " + " + ", ".join(others)
            if len(primary) + len(suffix) <= max_name_len:
                primary += suffix
        return primary


def _ensure_navec_downloaded() -> Path:
    """Download navec model with wget."""
    import subprocess

    _NAVEC_HOME.mkdir(parents=True, exist_ok=True)
    dest = _NAVEC_HOME / _NAVEC_MODEL

    if dest.exists():
        return dest

    # Try Python urllib first (no external dependency)
    try:
        from urllib.request import urlretrieve

        urlretrieve(_NAVEC_URL, str(dest))
    except Exception:
        # Fall back to wget
        try:
            subprocess.run(
                ["wget", "-O", str(dest), _NAVEC_URL],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to curl
            subprocess.run(
                ["curl", "-L", "-o", str(dest), _NAVEC_URL],
                check=True,
                capture_output=True,
            )

    return dest
