"""Differentiating skill selection for cross-specialty comparison.

Identifies the most differentiating skills across multiple specialties
using a TF-IDF-inspired scoring method. Used as a pure data pipeline
— no chart rendering, no database access.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from krm.analysis.compare import SkillComparator


def collect_differentiating_skills(
    skill_frequencies: Dict[str, Dict[str, int]],
    specialties: List[str],
    top_n: int = 8,
) -> List[Tuple[str, float, str]]:
    """Collect the most differentiating skills across all specialty pairs.

    For each pair of specialties (A, B), uses :class:`SkillComparator` to
    compute differentiating scores via ``freq_A / (1 + freq_B)``. A skill
    differentiates specialty A from B when it is prominent in A but rare
    or absent in B.

    The highest score across all pairs for each skill is retained, along
    with the specialty that the skill differentiates (its "home" specialty).

    Args:
        skill_frequencies: Nested dict mapping ``specialty → {skill_lemma → count}``.
            Count represents how many vacancies mention the skill for that specialty.
        specialties: List of specialty keys to compare. Must have at least 2 entries.
        top_n: Number of differentiating skills to return. Defaults to 8.

    Returns:
        List of ``(skill_lemma, score, from_specialty)`` tuples sorted by
        differentiation score descending. Score is ``freq_A / (1 + freq_B)``
        — higher means more distinctive. ``from_specialty`` identifies the
        specialty the skill originates from.

        Returns an empty list if fewer than 2 specialties are provided or
        no differentiating skills are found.

    Example:
        >>> freqs = {
        ...     "physics": {"python": 150, "xrd": 200, "docker": 60},
        ...     "chem":    {"python": 40,  "xrd": 80,  "синтез": 180},
        ... }
        >>> collect_differentiating_skills(freqs, ["physics", "chem"], top_n=3)
        [('синтез', 180.0, 'chem'), ('xrd', 66.6667, 'physics'), ('python', 37.5, 'physics')]
    """
    if len(specialties) < 2:
        return []

    comparator = SkillComparator(skill_frequencies)

    # Track the best score for each skill and which specialty it belongs to.
    # Key: skill lemma → max score seen so far.
    skill_best_score: Dict[str, float] = {}
    # Key: skill lemma → specialty where the skill had its best score.
    skill_source: Dict[str, str] = {}

    for i in range(len(specialties)):
        for j in range(i + 1, len(specialties)):
            spec_a = specialties[i]
            spec_b = specialties[j]

            diff = comparator.differentiating_skills(spec_a, spec_b, top_n=20)

            # Skills that differentiate specialty A from B
            for lemma, score in diff["a_vs_b"]:
                if score > skill_best_score.get(lemma, 0.0):
                    skill_best_score[lemma] = score
                    skill_source[lemma] = spec_a

            # Skills that differentiate specialty B from A
            for lemma, score in diff["b_vs_a"]:
                if score > skill_best_score.get(lemma, 0.0):
                    skill_best_score[lemma] = score
                    skill_source[lemma] = spec_b

    if not skill_best_score:
        return []

    # Sort by differentiation score descending, pick top N
    sorted_skills = sorted(
        skill_best_score.items(), key=lambda x: x[1], reverse=True
    )
    top_skills = sorted_skills[:top_n]

    return [
        (lemma, round(score, 4), skill_source[lemma])
        for lemma, score in top_skills
    ]
