"""Cross-specialty skill comparison: overlap analysis, uniqueness, differentiation.

Computes Jaccard similarity between skill sets and identifies differentiating
skills via a TF-IDF-inspired scoring method.
"""

from __future__ import annotations

from hh_competency.storage.db import Database
from hh_competency.storage.models import SkillFrequency


class SkillComparator:
    """Compare skill profiles across specialties.

    Uses cached skill frequencies from the database. Run SkillAnalyzer
    first to populate the cache for each specialty being compared.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def compare(
        self, specialties: list[str], top_n: int = 30
    ) -> dict[str, list[SkillFrequency]]:
        """Get top N skills for each specialty.

        Args:
            specialties: List of specialty keys to compare.
            top_n: Number of top skills per specialty.

        Returns:
            Dict mapping specialty → list of SkillFrequency (desc by frequency).
        """
        result: dict[str, list[SkillFrequency]] = {}
        for spec in specialties:
            result[spec] = self._db.get_skill_frequencies(spec, top_n=top_n)
        return result

    def unique_skills(
        self, specialty: str, vs_specialties: list[str], min_count: int = 3
    ) -> list[str]:
        """Find skills unique to one specialty compared to a set of others.

        A skill is unique if it appears in the target specialty (≥ min_count)
        and does NOT appear in any of the comparison specialties (at any frequency).

        Args:
            specialty: Target specialty to find unique skills for.
            vs_specialties: Comparison specialties.
            min_count: Minimum vacancy_count for the target specialty.

        Returns:
            List of unique skill lemmas sorted alphabetically.
        """
        target_skills = self._db.get_skill_frequencies(specialty, top_n=500)
        target_lemmas = {f.lemma for f in target_skills if f.vacancy_count >= min_count}

        # Collect all lemmas from comparison specialties
        other_lemmas: set[str] = set()
        for vs_spec in vs_specialties:
            vs_skills = self._db.get_skill_frequencies(vs_spec, top_n=500)
            other_lemmas.update(f.lemma for f in vs_skills)

        unique = target_lemmas - other_lemmas
        return sorted(unique)

    def shared_skills(
        self, specialties: list[str], min_count: int = 3
    ) -> list[str]:
        """Find skills shared across all specified specialties.

        Args:
            specialties: Specialty keys to intersect.
            min_count: Minimum vacancy_count for each specialty.

        Returns:
            List of shared skill lemmas sorted alphabetically.
        """
        if not specialties:
            return []

        skill_sets: list[set[str]] = []
        for spec in specialties:
            skills = self._db.get_skill_frequencies(spec, top_n=500)
            lemmas = {f.lemma for f in skills if f.vacancy_count >= min_count}
            skill_sets.append(lemmas)

        shared = skill_sets[0]
        for s in skill_sets[1:]:
            shared = shared & s

        return sorted(shared)

    def overlap_matrix(
        self, specialties: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise Jaccard similarity between specialty skill sets.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|, using all skills with frequency ≥ 1.

        Args:
            specialties: Specialty keys to compare.

        Returns:
            Nested dict: matrix[spec_a][spec_b] = Jaccard similarity (0.0–1.0).
            Diagonal entries (spec_a == spec_b) are 1.0.
        """
        # Load all skill sets
        lemma_sets: dict[str, set[str]] = {}
        for spec in specialties:
            skills = self._db.get_skill_frequencies(spec, top_n=1000)
            lemma_sets[spec] = {f.lemma for f in skills}

        matrix: dict[str, dict[str, float]] = {}
        for spec_a in specialties:
            matrix[spec_a] = {}
            set_a = lemma_sets.get(spec_a, set())
            for spec_b in specialties:
                if spec_a == spec_b:
                    matrix[spec_a][spec_b] = 1.0
                else:
                    set_b = lemma_sets.get(spec_b, set())
                    intersection = len(set_a & set_b)
                    union = len(set_a | set_b)
                    matrix[spec_a][spec_b] = (
                        intersection / union if union > 0 else 0.0
                    )

        return matrix

    def differentiating_skills(
        self, specialty_a: str, specialty_b: str, top_n: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        """Find skills that differentiate specialty A from B.

        Uses TF-IDF-inspired scoring: score = freq_A / (1 + freq_B).
        A high score means the skill is prominent in A but rare in B.

        Args:
            specialty_a: Primary specialty (skills scored for differentiation).
            specialty_b: Comparison specialty (skills in A that are rare here score high).
            top_n: Number of top differentiating skills per direction.

        Returns:
            Dict with keys 'a_vs_b' and 'b_vs_a', each mapping to a list of
            (skill_lemma, score) tuples sorted by score descending.
        """
        skills_a = self._db.get_skill_frequencies(specialty_a, top_n=500)
        skills_b = self._db.get_skill_frequencies(specialty_b, top_n=500)

        freq_a: dict[str, int] = {f.lemma: f.frequency for f in skills_a}
        freq_b: dict[str, int] = {f.lemma: f.frequency for f in skills_b}

        def _score(
            primary: dict[str, int], other: dict[str, int]
        ) -> list[tuple[str, float]]:
            scores: list[tuple[str, float]] = []
            for lemma, f_primary in primary.items():
                f_other = other.get(lemma, 0)
                score = f_primary / (1.0 + f_other)
                scores.append((lemma, round(score, 4)))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_n]

        return {
            "a_vs_b": _score(freq_a, freq_b),
            "b_vs_a": _score(freq_b, freq_a),
        }
