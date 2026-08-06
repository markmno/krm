"""Mixed specialty model builder: synthesize hybrid competency models.

When a specialty has `mix_of: [parent_a, parent_b]`, this module
combines parent skill frequencies and role profiles to produce a
hybrid competency model for the composite specialty (e.g.
bioinformatics = biology ∩ IT, physicist-engineer = physics + engineering).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from hh_competency.storage.db import Database
from hh_competency.storage.models import AppConfig, RoleProfile, SkillFrequency


@dataclass
class MixedSkillProfile:
    """Combined skill profile for a mixed specialty."""

    specialty: str
    parents: list[str]
    combined_frequencies: list[SkillFrequency] = field(default_factory=list)
    intersection_frequencies: list[SkillFrequency] = field(
        default_factory=list
    )
    uniqueness_map: dict[str, list[str]] = field(default_factory=dict)
    merged_roles: list[RoleProfile] = field(default_factory=list)
    # Metrics
    jaccard_similarity: float = 0.0
    total_unique_skills: int = 0  # union count
    shared_skill_count: int = 0
    source_distribution: dict[str, int] = field(
        default_factory=dict
    )  # parent → count of dominated skills


class MixedModelBuilder:
    """Builds hybrid competency models by combining parent specialty data.

    Three merging strategies:
    1. UNION — all skills from all parents (broad model, e.g. stem+engineering)
    2. INTERSECTION — skills present in ALL parents (focused, e.g. biophysics core)
    3. WEIGHTED — skills scored by presence across parents (e.g. stem+it)
    """

    def __init__(self, db: Database, config: AppConfig) -> None:
        self._db = db
        self._config = config

    def get_mixable_specialties(self) -> list[str]:
        """Return specialty names that have mix_of defined."""
        return [
            name
            for name, spec in self._config.specialties.items()
            if spec.mix_of
        ]

    def build_mixed_profile(
        self, mixed_name: str, strategy: str = "union"
    ) -> MixedSkillProfile:
        """Build a hybrid skill profile for a mixed specialty.

        Args:
            mixed_name: Name of the mixed specialty (must have mix_of).
            strategy: Merging strategy — 'union', 'intersection', or 'weighted'.

        Returns:
            MixedSkillProfile with combined frequencies, roles, uniqueness map.
        """
        spec = self._config.specialties.get(mixed_name)
        if not spec or not spec.mix_of:
            raise ValueError(f"'{mixed_name}' is not a mixable specialty")

        parents = spec.mix_of

        # Load parent skill frequencies
        parent_skills: dict[str, list[SkillFrequency]] = {}
        for parent in parents:
            freqs = self._db.get_skill_frequencies(parent, top_n=200)
            if freqs:
                parent_skills[parent] = freqs

        if len(parent_skills) < 2:
            # Only one parent has data — return its profile as-is
            return self._single_parent_profile(
                mixed_name, parents, parent_skills
            )

        # Build combined frequency lists
        if strategy == "intersection":
            combined = self._merge_intersection(mixed_name, parent_skills)
            total = len(combined)
        elif strategy == "weighted":
            combined, total = self._merge_weighted(
                mixed_name, parent_skills
            )
        else:  # union (default)
            combined, total = self._merge_union(mixed_name, parent_skills)

        # Compute uniqueness map: which skills are unique to which parent
        uniqueness = self._compute_uniqueness(parent_skills)

        # Build intersection set
        intersection = self._merge_intersection(
            mixed_name, parent_skills
        )

        # Compute Jaccard similarity between parent skill sets
        jaccard = self._compute_parent_jaccard(parent_skills, total)

        # Merge role profiles
        merged_roles = self._merge_roles(mixed_name, parents)

        # Source distribution: which parent dominates the combined profile
        source_dist = self._compute_source_distribution(
            combined, parent_skills
        )

        return MixedSkillProfile(
            specialty=mixed_name,
            parents=parents,
            combined_frequencies=combined,
            intersection_frequencies=intersection,
            uniqueness_map=uniqueness,
            merged_roles=merged_roles,
            jaccard_similarity=jaccard,
            total_unique_skills=total,
            shared_skill_count=len(intersection),
            source_distribution=source_dist,
        )

    def build_all_mixed(self) -> dict[str, MixedSkillProfile]:
        """Build mixed profiles for all specialties with mix_of."""
        results: dict[str, MixedSkillProfile] = {}
        for name in self.get_mixable_specialties():
            try:
                results[name] = self.build_mixed_profile(name)
            except Exception:
                results[name] = MixedSkillProfile(
                    specialty=name,
                    parents=self._config.specialties[name].mix_of or [],
                )
                results[name].combined_frequencies = []
        return results

    # ── Internal merge strategies ──

    def _load_parent_skills(
        self, parents: list[str]
    ) -> dict[str, list[SkillFrequency]]:
        result: dict[str, list[SkillFrequency]] = {}
        for parent in parents:
            freqs = self._db.get_skill_frequencies(parent, top_n=200)
            if freqs:
                result[parent] = freqs
        return result

    @staticmethod
    def _single_parent_profile(
        mixed_name: str,
        parents: list[str],
        parent_skills: dict[str, list[SkillFrequency]],
    ) -> MixedSkillProfile:
        """Fallback: only one parent has data."""
        for parent_freqs in parent_skills.values():
            return MixedSkillProfile(
                specialty=mixed_name,
                parents=parents,
                combined_frequencies=parent_freqs,
                intersection_frequencies=parent_freqs,
                uniqueness_map={},
                total_unique_skills=len(parent_freqs),
                shared_skill_count=len(parent_freqs),
                source_distribution={parents[0]: len(parent_freqs)},
            )
        return MixedSkillProfile(specialty=mixed_name, parents=parents)

    @staticmethod
    def _merge_union(
        mixed_name: str,
        parent_skills: dict[str, list[SkillFrequency]],
    ) -> tuple[list[SkillFrequency], int]:
        """Union: all skills from all parents, frequency = sum of parent frequencies."""
        merged: Counter[str] = Counter()
        lemma_meta: dict[str, dict[str, Any]] = {}
        for parent_freqs in parent_skills.values():
            for sf in parent_freqs:
                merged[sf.lemma] += sf.frequency
                if sf.lemma not in lemma_meta:
                    lemma_meta[sf.lemma] = {
                        "pos": sf.pos,
                        "vacancy_count": sf.vacancy_count,
                    }
                else:
                    lemma_meta[sf.lemma]["vacancy_count"] += sf.vacancy_count

        result = [
            SkillFrequency(
                specialty=mixed_name,
                lemma=lemma,
                pos=lemma_meta[lemma]["pos"],
                frequency=freq,
                vacancy_count=lemma_meta[lemma]["vacancy_count"],
            )
            for lemma, freq in merged.most_common()
        ]
        return result, len(result)

    @staticmethod
    def _merge_intersection(
        mixed_name: str,
        parent_skills: dict[str, list[SkillFrequency]],
    ) -> list[SkillFrequency]:
        """Intersection: only skills present in ALL parents."""
        if not parent_skills:
            return []

        parent_lemma_sets = [
            frozenset(sf.lemma for sf in freqs)
            for freqs in parent_skills.values()
        ]
        common = parent_lemma_sets[0]
        for s in parent_lemma_sets[1:]:
            common = common & s

        # Build frequencies for common skills (avg frequency across parents)
        result: list[SkillFrequency] = []
        for lemma in common:
            freqs = [
                next(
                    (sf.frequency for sf in pf if sf.lemma == lemma), 0
                )
                for pf in parent_skills.values()
            ]
            avg_freq = int(sum(freqs) / len(freqs))
            pos = next(
                (sf.pos for pf in parent_skills.values() for sf in pf if sf.lemma == lemma),
                "NOUN",
            )
            result.append(
                SkillFrequency(
                    specialty=mixed_name,
                    lemma=lemma,
                    pos=pos,
                    frequency=avg_freq,
                    vacancy_count=sum(freqs),
                )
            )

        result.sort(key=lambda x: x.frequency, reverse=True)
        return result

    @staticmethod
    def _merge_weighted(
        mixed_name: str,
        parent_skills: dict[str, list[SkillFrequency]],
    ) -> tuple[list[SkillFrequency], int]:
        """Weighted: skills scored by how many parents they appear in.

        Score = (n_parents_with_skill / total_parents) * sum_of_frequencies.
        Skills in all parents get full weight; single-parent skills are penalized.
        """
        n_parents = len(parent_skills)
        lemma_scores: Counter[str] = Counter()
        lemma_info: dict[str, dict[str, Any]] = {}

        for pf in parent_skills.values():
            for sf in pf:
                lemma_scores[sf.lemma] += 1  # parent count
                if sf.lemma not in lemma_info:
                    lemma_info[sf.lemma] = {
                        "freq_sum": sf.frequency,
                        "pos": sf.pos,
                        "vc_sum": sf.vacancy_count,
                    }
                else:
                    lemma_info[sf.lemma]["freq_sum"] += sf.frequency
                    lemma_info[sf.lemma]["vc_sum"] += sf.vacancy_count

        # Weighted score = presence ratio * total frequency
        weighted: list[tuple[str, float]] = []
        for lemma, parent_count in lemma_scores.items():
            presence_ratio = parent_count / n_parents
            score = presence_ratio * lemma_info[lemma]["freq_sum"]
            weighted.append((lemma, score))

        weighted.sort(key=lambda x: x[1], reverse=True)

        result = [
            SkillFrequency(
                specialty=mixed_name,
                lemma=lemma,
                pos=lemma_info[lemma]["pos"],
                frequency=int(score),
                vacancy_count=lemma_info[lemma]["vc_sum"],
            )
            for lemma, score in weighted
        ]
        return result, len(result)

    @staticmethod
    def _compute_uniqueness(
        parent_skills: dict[str, list[SkillFrequency]],
    ) -> dict[str, list[str]]:
        """Find skills unique to each parent (not shared with any other parent)."""
        if len(parent_skills) < 2:
            return {}

        parent_lemma_sets = {
            name: frozenset(sf.lemma for sf in freqs)
            for name, freqs in parent_skills.items()
        }

        uniqueness: dict[str, list[str]] = {}
        all_parents = list(parent_skills.keys())
        for parent, lemma_set in parent_lemma_sets.items():
            others = [
                parent_lemma_sets[p]
                for p in all_parents
                if p != parent
            ]
            other_union: frozenset[str] = frozenset()
            for s in others:
                other_union |= s
            unique = sorted(lemma_set - other_union)
            uniqueness[parent] = unique

        return uniqueness

    @staticmethod
    def _compute_parent_jaccard(
        parent_skills: dict[str, list[SkillFrequency]],
        union_size: int,
    ) -> float:
        """Jaccard similarity between parent skill sets."""
        if len(parent_skills) < 2 or union_size == 0:
            return 0.0

        all_sets = [
            frozenset(sf.lemma for sf in freqs)
            for freqs in parent_skills.values()
        ]
        intersection = all_sets[0]
        for s in all_sets[1:]:
            intersection = intersection & s

        return len(intersection) / union_size if union_size > 0 else 0.0

    def _merge_roles(
        self, mixed_name: str, parents: list[str]
    ) -> list[RoleProfile]:
        """Merge role profiles from parent specialties.

        Strategy: collect all parent roles, prefix with parent name for
        disambiguation, sort by parent.
        """
        merged: list[RoleProfile] = []
        for parent in parents:
            try:
                parent_roles = self._db.get_role_profiles(parent)
                for role in parent_roles:
                    merged.append(
                        RoleProfile(
                            specialty=mixed_name,
                            role_name=f"[{parent}] {role.role_name}",
                            defining_skills=role.defining_skills,
                            supporting_skills=role.supporting_skills,
                        )
                    )
            except Exception:
                continue
        return merged

    @staticmethod
    def _compute_source_distribution(
        combined: list[SkillFrequency],
        parent_skills: dict[str, list[SkillFrequency]],
    ) -> dict[str, int]:
        """Count how many combined skills originate from each parent (dominated by)."""
        distribution: Counter[str] = Counter()

        # For each combined skill, find which parent has the highest frequency
        parent_lemma_strength: dict[
            str, dict[str, int]
        ] = {}  # lemma → {parent: freq}
        for parent, freqs in parent_skills.items():
            for sf in freqs:
                if sf.lemma not in parent_lemma_strength:
                    parent_lemma_strength[sf.lemma] = {}
                parent_lemma_strength[sf.lemma][parent] = sf.frequency

        for sf in combined:
            strengths = parent_lemma_strength.get(sf.lemma, {})
            if strengths:
                dominant = max(strengths, key=lambda k: strengths[k])
                distribution[dominant] += 1

        return dict(distribution)
