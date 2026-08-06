"""Role discovery via skill co-occurrence clustering.

Builds a skill×vacancy co-occurrence matrix, clusters skills using
AgglomerativeClustering with Jaccard distance, and names the resulting
clusters as role profiles (e.g. "Python бэкенд разработчик").
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from hh_competency.nlp.pipeline import NLPPipeline
from hh_competency.storage.db import Database
from hh_competency.storage.models import RoleProfile


class SkillClusterer:
    """Discovers role profiles from clustered skill co-occurrence patterns.

    Pipeline:
    1. Build binary skill×vacancy co-occurrence matrix
    2. Compute Jaccard distance between skills
    3. Agglomerative clustering of skills
    4. Filter small clusters, name larger ones → RoleProfile
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

    def discover_roles(
        self,
        specialty: str,
        n_clusters: int = 3,
        min_cluster_size: int = 5,
    ) -> list[RoleProfile]:
        """Discover competency role profiles from skill co-occurrence clusters.

        Args:
            specialty: Specialty key (e.g. 'data_science').
            n_clusters: Target number of clusters for AgglomerativeClustering.
            min_cluster_size: Drop clusters with fewer than this many skills.

        Returns:
            List of RoleProfile, one per cluster meeting min_cluster_size.
            Results are persisted to DB via insert_role_profiles.
        """
        skill_list, matrix = self._build_cooccurrence_matrix(specialty)

        if len(skill_list) < min_cluster_size * n_clusters:
            return []

        freq_data = self._db.get_skill_frequencies(specialty, top_n=len(skill_list))
        lemma_to_freq: dict[str, int] = {f.lemma: f.frequency for f in freq_data}

        clusters = self._cluster_skills(skill_list, matrix, lemma_to_freq, n_clusters)

        profiles: list[RoleProfile] = []
        for skills_with_weights in clusters:
            if len(skills_with_weights) < min_cluster_size:
                continue
            role_name = self._name_cluster(skills_with_weights)
            defining = skills_with_weights[:5]
            supporting = skills_with_weights[5:]
            profiles.append(
                RoleProfile(
                    specialty=specialty,
                    role_name=role_name,
                    defining_skills=defining,
                    supporting_skills=supporting,
                )
            )

        if profiles:
            self._db.insert_role_profiles(profiles)

        return profiles

    # -- Private helpers --

    def _build_cooccurrence_matrix(
        self, specialty: str
    ) -> tuple[list[str], np.ndarray]:
        """Build binary skill×vacancy co-occurrence matrix.

        Each row corresponds to a skill (lemma), each column to a vacancy.
        Entry [i, j] = 1 if skill i appears in vacancy j.

        Args:
            specialty: Specialty key.

        Returns:
            Tuple of (skill_list, cooccurrence_matrix).
            skill_list[i] is the lemma for row i of the matrix.
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

    def _cluster_skills(
        self,
        skills: list[str],
        matrix: np.ndarray,
        freq_map: dict[str, int],
        n_clusters: int,
    ) -> list[list[tuple[str, float]]]:
        """Cluster skills using AgglomerativeClustering with Jaccard distance.

        Args:
            skills: Skill lemma list (aligned with matrix rows).
            matrix: Binary skill×vacancy co-occurrence matrix (n_skills × n_vacancies).
            freq_map: Lemma → frequency map for computing skill weights.
            n_clusters: Target number of clusters.

        Returns:
            List of clusters, each a list of (lemma, weight) sorted by weight desc.
            Sorted by cluster size (largest first).
        """
        n_skills = len(skills)
        if n_skills == 0:
            return []

        n_actual = min(n_clusters, n_skills)

        # Compute Jaccard distance matrix between skill vectors
        if n_skills > 1:
            dist_condensed = pdist(matrix, metric="jaccard")
            dist_matrix = squareform(dist_condensed)
        else:
            # Single skill — trivial cluster
            lemma = skills[0]
            weight = float(freq_map.get(lemma, 0))
            return [[(lemma, weight)]]

        clustering = AgglomerativeClustering(
            n_clusters=n_actual,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)

        # Group skills by cluster label
        groups: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for idx, label in enumerate(labels):
            lemma = skills[idx]
            weight = float(freq_map.get(lemma, 0))
            groups[label].append((lemma, weight))

        # Sort within each cluster by weight desc
        for label in groups:
            groups[label].sort(key=lambda x: x[1], reverse=True)

        # Sort clusters by size (descending) for consistent ordering
        sorted_clusters = sorted(groups.values(), key=len, reverse=True)
        return sorted_clusters

    @staticmethod
    def _name_cluster(skills_with_weights: list[tuple[str, float]]) -> str:
        """Generate a role name from the cluster's skill categories.

        Uses _classify_skill_to_category from axis_discovery to map skills
        to competency categories, then picks the dominant category.
        Falls back to top-3 skill concatenation if no category matches.

        Args:
            skills_with_weights: Sorted list of (lemma, weight) for the cluster.

        Returns:
            Role name string (e.g. 'Программирование и IT: ML/AI specialist').
        """
        from collections import Counter

        from hh_competency.analysis.axis_discovery import _classify_skill_to_category

        skill_lemmas = [lemma for lemma, _w in skills_with_weights[:15]]
        category_counts: Counter[str] = Counter()
        categorized: dict[str, list[str]] = {}

        for lemma in skill_lemmas:
            cat = _classify_skill_to_category(lemma)
            if cat:
                category_counts[cat] += 1
                categorized.setdefault(cat, []).append(lemma)

        if category_counts:
            dominant = category_counts.most_common(1)[0][0]
            top_skills = skill_lemmas[:3]
            num_classified = sum(category_counts.values())

            if num_classified >= 3 and len(category_counts) == 1:
                # All skills in one category — concise name
                return dominant

            if num_classified >= 2:
                if len(category_counts) >= 2:
                    # Two categories — compound name
                    cats = [c for c, _ in category_counts.most_common(2)]
                    return f"{cats[0]} / {cats[1]}"
                # One dominant category — name + top skills
                return f"{dominant}: {', '.join(top_skills[:2])}"

            # Classified but sparse — category + top skill
            return f"{dominant}: {top_skills[0]}"

        # No category match — fallback to raw concatenation
        return " ".join(skill_lemmas[:3])
