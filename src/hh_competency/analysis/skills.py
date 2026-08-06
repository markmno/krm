"""Skill frequency extraction and ranking for vacancy text analysis.

Uses NLPPipeline (razdel + pymorphy3) to lemmatize descriptions,
then counts lemma occurrences per document with vacancy_count tracking.
"""

from __future__ import annotations

from hh_competency.nlp.pipeline import NLPPipeline
from hh_competency.storage.db import Database
from hh_competency.storage.models import SkillFrequency


class NoDataError(Exception):
    """Raised when no vacancy data exists for a specialty."""


class SkillAnalyzer:
    """Extracts and ranks skills from vacancy descriptions.

    Combines NLPPipeline extraction with DuckDB persistence:
    raw descriptions → lemmatization → frequency counting → DB cache.
    """

    def __init__(self, db: Database, pipeline: NLPPipeline) -> None:
        self._db = db
        self._pipeline = pipeline

    def compute_frequencies(
        self, specialty: str, min_count: int = 2
    ) -> list[SkillFrequency]:
        """Extract lemma frequencies from all vacancy descriptions for a specialty.

        Pipeline per vacancy:
        1. razdel tokenize → pymorphy3 lemmatize → stopword filter → POS filter
        2. Collect unique lemmas per vacancy for vacancy_count tracking
        3. Aggregate across all vacancies

        Args:
            specialty: Specialty key (e.g. 'data_science', 'python_dev').
            min_count: Minimum vacancy count threshold (lemmas in fewer
                       vacancies are discarded).

        Returns:
            SkillFrequency list sorted by frequency descending.

        Raises:
            NoDataError: If no descriptions exist for the specialty.
        """
        descriptions = self._db.get_vacancy_descriptions(specialty)
        if not descriptions:
            raise NoDataError(
                f"No vacancy descriptions found for specialty: {specialty}"
            )

        # lemma → dict with pos and vacancy_count
        lemma_data: dict[str, dict] = {}

        for desc in descriptions:
            keywords = self._pipeline.extract_keywords(desc)
            seen: set[str] = set()
            for lemma, pos in keywords:
                if lemma not in lemma_data:
                    lemma_data[lemma] = {"pos": pos, "vacancy_count": 0}
                if lemma not in seen:
                    lemma_data[lemma]["vacancy_count"] += 1
                    seen.add(lemma)

        # Build result list, filtering by min_count
        results: list[SkillFrequency] = []
        for lemma, data in lemma_data.items():
            if data["vacancy_count"] >= min_count:
                results.append(
                    SkillFrequency(
                        specialty=specialty,
                        lemma=lemma,
                        pos=data["pos"],
                        frequency=data["vacancy_count"],
                        vacancy_count=data["vacancy_count"],
                    )
                )

        results.sort(key=lambda f: f.frequency, reverse=True)

        if results:
            self._db.insert_skill_frequencies(results)

        return results

    def top_skills(self, specialty: str, n: int = 50) -> list[SkillFrequency]:
        """Get top N skills for a specialty from DB cache.

        If no cached frequencies exist, runs compute_frequencies first.

        Args:
            specialty: Specialty key.
            n: Number of top skills to return.

        Returns:
            SkillFrequency list sorted by frequency descending.
        """
        cached = self._db.get_skill_frequencies(specialty, top_n=n)
        if cached:
            return cached
        # Nothing cached — compute from scratch
        all_skills = self.compute_frequencies(specialty)
        return all_skills[:n]
