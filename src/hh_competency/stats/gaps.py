"""Curriculum-market gap analysis for competency models.

Compares taught skills (curriculum) against market-demanded skills to identify
coverage gaps, excess training, and Jaccard similarity between what is taught
and what the labor market requires.
"""

from __future__ import annotations

from hh_competency.storage.db import Database


class GapAnalyzer:
    """Analyzes gaps between curriculum (taught skills) and market demand.

    Uses database-cached skill frequencies as the market reference and
    external curriculum skill lists as the comparison target. Computes
    coverage percentages, missing/excess skills, and Jaccard gap metrics.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- Public API --

    def curriculum_market_gap(
        self,
        curriculum_skills: list[str],
        specialty: str,
    ) -> dict:
        """Compare a curriculum's taught skills against market demand.

        Identifies which market-demanded skills are missing from the
        curriculum and which taught skills have no market demand.

        Args:
            curriculum_skills: List of skill lemmas taught in the curriculum.
            specialty: Specialty key (e.g. 'data_science').

        Returns:
            Dict with keys:
            - missing_skills (list[str]): Market skills not in curriculum,
              sorted by frequency (most in-demand first).
            - excess_skills (list[str]): Curriculum skills not in market.
            - coverage_pct (float): Percentage of market skills covered by
              curriculum (0–100).
            - jaccard_gap (float): 1 - Jaccard similarity, where 1 = no overlap,
              0 = perfect overlap.
            - market_skill_count (int): Total unique market skills considered.
            - curriculum_skill_count (int): Total unique curriculum skills.
            - matched_skill_count (int): Number of skills in both sets.
        """
        market_freqs = self._db.get_skill_frequencies(specialty, top_n=500)
        market_lemmas: set[str] = {f.lemma for f in market_freqs}
        curriculum_set: set[str] = set(curriculum_skills)

        # Missing: in market but not in curriculum
        missing_in_market_order = [
            f.lemma for f in market_freqs if f.lemma not in curriculum_set
        ]
        # Excess: in curriculum but not in market
        excess = sorted(curriculum_set - market_lemmas)
        # Matched: in both
        matched = market_lemmas & curriculum_set

        # Coverage percentage
        market_count = len(market_lemmas)
        matched_count = len(matched)
        coverage_pct = _safe_pct(matched_count, market_count)

        # Jaccard similarity and gap
        union = market_lemmas | curriculum_set
        jaccard_sim = len(matched) / len(union) if union else 0.0
        jaccard_gap = 1.0 - jaccard_sim

        return {
            "missing_skills": missing_in_market_order,
            "excess_skills": excess,
            "coverage_pct": round(coverage_pct, 1),
            "jaccard_gap": round(jaccard_gap, 4),
            "market_skill_count": market_count,
            "curriculum_skill_count": len(curriculum_set),
            "matched_skill_count": matched_count,
        }

    def skill_coverage_matrix(
        self,
        specialties: list[str],
        curriculum_map: dict[str, list[str]],
    ) -> dict:
        """Compute coverage stats for each specialty→curriculum pair.

        For a set of specialties and corresponding curriculum skill lists,
        computes missing, excess, coverage, and Jaccard metrics for every
        pairing — including cross-pairings where specialty ≠ curriculum key.

        Args:
            specialties: List of specialty keys for market data.
            curriculum_map: Dict mapping a curriculum name to its taught
              skill list. Keys may or may not match specialty names.

        Returns:
            Dict with keys:
            - matrix (list[list[dict]]): 2D list of per-pair result dicts,
              rows = specialties, cols = curriculum keys.
            - row_labels (list[str]): Specialty names (row order).
            - col_labels (list[str]): Curriculum names (column order).
            - summary (dict): Aggregated stats:
              * mean_coverage_pct: average coverage across all pairs.
              * best_pair: (specialty, curriculum) with highest coverage.
              * worst_pair: (specialty, curriculum) with lowest coverage.
        """
        if not specialties or not curriculum_map:
            return _empty_coverage_matrix()

        curric_keys = list(curriculum_map.keys())
        matrix: list[list[dict]] = []

        best_coverage = -1.0
        best_pair: tuple[str, str] | None = None
        worst_coverage = 101.0
        worst_pair: tuple[str, str] | None = None
        all_coverages: list[float] = []

        for spec in specialties:
            row: list[dict] = []
            for curric_key in curric_keys:
                result = self.curriculum_market_gap(
                    curriculum_skills=curriculum_map[curric_key],
                    specialty=spec,
                )
                # Slim down result for matrix display
                cell = {
                    "specialty": spec,
                    "curriculum": curric_key,
                    "coverage_pct": result["coverage_pct"],
                    "jaccard_gap": result["jaccard_gap"],
                    "missing_count": len(result["missing_skills"]),
                    "excess_count": len(result["excess_skills"]),
                    "matched_skill_count": result["matched_skill_count"],
                }
                row.append(cell)

                cov = result["coverage_pct"]
                all_coverages.append(cov)
                if cov > best_coverage:
                    best_coverage = cov
                    best_pair = (spec, curric_key)
                if cov < worst_coverage:
                    worst_coverage = cov
                    worst_pair = (spec, curric_key)

            matrix.append(row)

        # Summary
        mean_cov = (
            sum(all_coverages) / len(all_coverages) if all_coverages else 0.0
        )

        return {
            "matrix": matrix,
            "row_labels": list(specialties),
            "col_labels": curric_keys,
            "summary": {
                "mean_coverage_pct": round(mean_cov, 1),
                "best_pair": list(best_pair) if best_pair else [],
                "worst_pair": list(worst_pair) if worst_pair else [],
                "best_coverage_pct": round(best_coverage, 1) if best_coverage >= 0 else 0.0,
                "worst_coverage_pct": round(worst_coverage, 1) if worst_coverage <= 100 else 0.0,
            },
        }


# -- Helpers --


def _safe_pct(numerator: int, denominator: int) -> float:
    """Compute percentage safely, returning 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _empty_coverage_matrix() -> dict:
    return {
        "matrix": [],
        "row_labels": [],
        "col_labels": [],
        "summary": {
            "mean_coverage_pct": 0.0,
            "best_pair": [],
            "worst_pair": [],
            "best_coverage_pct": 0.0,
            "worst_coverage_pct": 0.0,
        },
    }
