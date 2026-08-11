"""Cross-specialty skill comparison: overlap analysis, uniqueness, differentiation.

Computes Jaccard similarity between skill sets and identifies differentiating
skills via a TF-IDF-inspired scoring method.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class SkillComparator:
    """Compare skill profiles across specialties.

    Uses pre-computed skill frequency data. The caller is responsible for
    building the frequency dict (e.g. from a database or file), then passing
    it to the comparator.

    Attributes:
        _freqs: Nested dict mapping specialty → {skill_lemma → count}.
    """

    def __init__(self, skill_frequencies: Dict[str, Dict[str, int]]) -> None:
        """Initialize with pre-computed skill frequency data.

        Args:
            skill_frequencies: Mapping from specialty key to a dict of
                {skill_lemma: frequency_count}. The count represents how many
                vacancies mention the skill for that specialty.
        """
        self._freqs = skill_frequencies

    def compare(
        self, specialties: List[str], top_n: int = 30
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Get top N skills for each specialty.

        Args:
            specialties: List of specialty keys to compare.
            top_n: Number of top skills per specialty.

        Returns:
            Dict mapping specialty → list of (skill_lemma, count) sorted
            by count descending.
        """
        result: Dict[str, List[Tuple[str, int]]] = {}
        for spec in specialties:
            freq = self._freqs.get(spec, {})
            sorted_skills = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            result[spec] = sorted_skills[:top_n]
        return result

    def unique_skills(
        self,
        specialty: str,
        vs_specialties: List[str],
        min_count: int = 3,
    ) -> List[str]:
        """Find skills unique to one specialty compared to a set of others.

        A skill is unique if it appears in the target specialty (≥ min_count)
        and does NOT appear in any of the comparison specialties (at any
        frequency).

        Args:
            specialty: Target specialty to find unique skills for.
            vs_specialties: List of comparison specialties.
            min_count: Minimum frequency count for the target specialty.

        Returns:
            List of unique skill lemmas sorted alphabetically.
        """
        target_freq = self._freqs.get(specialty, {})
        target_lemmas = {
            lemma for lemma, count in target_freq.items() if count >= min_count
        }

        # Collect all lemmas from comparison specialties
        other_lemmas: set[str] = set()
        for vs_spec in vs_specialties:
            vs_freq = self._freqs.get(vs_spec, {})
            other_lemmas.update(vs_freq.keys())

        unique = target_lemmas - other_lemmas
        return sorted(unique)

    def shared_skills(
        self, specialties: List[str], min_count: int = 3
    ) -> List[str]:
        """Find skills shared across all specified specialties.

        Args:
            specialties: Specialty keys to intersect.
            min_count: Minimum frequency count for each specialty.

        Returns:
            List of shared skill lemmas sorted alphabetically.
        """
        if not specialties:
            return []

        skill_sets: List[set[str]] = []
        for spec in specialties:
            freq = self._freqs.get(spec, {})
            lemmas = {
                lemma for lemma, count in freq.items() if count >= min_count
            }
            skill_sets.append(lemmas)

        shared = skill_sets[0]
        for s in skill_sets[1:]:
            shared = shared & s

        return sorted(shared)

    def overlap_matrix(
        self, specialties: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Jaccard similarity between specialty skill sets.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|, using all skills with frequency ≥ 1.

        Args:
            specialties: Specialty keys to compare.

        Returns:
            Nested dict: matrix[spec_a][spec_b] = Jaccard similarity (0.0–1.0).
            Diagonal entries (spec_a == spec_b) are 1.0.
        """
        # Load all skill sets
        lemma_sets: Dict[str, set[str]] = {}
        for spec in specialties:
            freq = self._freqs.get(spec, {})
            lemma_sets[spec] = set(freq.keys())

        matrix: Dict[str, Dict[str, float]] = {}
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
        self,
        specialty_a: str,
        specialty_b: str,
        top_n: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Find skills that differentiate specialty A from B.

        Uses TF-IDF-inspired scoring: score = freq_A / (1 + freq_B).
        A high score means the skill is prominent in A but rare in B.

        Args:
            specialty_a: Primary specialty (skills scored for differentiation).
            specialty_b: Comparison specialty (skills in A that are rare here score high).
            top_n: Number of top differentiating skills per direction.

        Returns:
            Dict with keys ``'a_vs_b'`` and ``'b_vs_a'``, each mapping to a
            list of (skill_lemma, score) tuples sorted by score descending.
        """
        freq_a = self._freqs.get(specialty_a, {})
        freq_b = self._freqs.get(specialty_b, {})

        def _score(
            primary: Dict[str, int], other: Dict[str, int]
        ) -> List[Tuple[str, float]]:
            scores: List[Tuple[str, float]] = []
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
