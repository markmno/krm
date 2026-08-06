"""FGOS comparator: compares market-derived competencies with FGOS educational standards.

Identifies curriculum gaps: what FGOS requires but market doesn't need,
and what market demands but FGOS doesn't teach.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hh_competency.storage.db import Database

# -- FGOS educational standard templates --
FGOS_TEMPLATES: dict[str, dict] = {
    "03.03.02": {
        "название": "Физика",
        "компетенции": [
            "способность использовать базовые теоретические знания",
            "способность применять методы физического эксперимента",
            "способность использовать современные информационные технологии",
            "способность к анализу и обобщению научной информации",
            "способность применять математический аппарат",
            "способность использовать специализированное оборудование",
            "способность оформлять результаты исследований",
            "способность к педагогической деятельности",
        ],
    },
    "04.03.01": {
        "название": "Химия",
        "компетенции": [
            "способность использовать теоретические основы химии",
            "способность проводить химический эксперимент",
            "способность использовать стандартное лабораторное оборудование",
            "способность к анализу химических данных",
            "способность применять методы математической статистики",
            "способность оформлять научно-техническую документацию",
            "способность работать с научной литературой",
            "способность к педагогической деятельности",
        ],
    },
    "06.03.01": {
        "название": "Биология",
        "компетенции": [
            "способность применять знания в области биологии",
            "способность проводить биологический эксперимент",
            "способность использовать лабораторное оборудование",
            "способность к анализу и статистической обработке данных",
            "способность к полевой и лабораторной работе",
            "способность оформлять результаты исследований",
            "способность использовать информационные технологии",
            "способность к педагогической деятельности",
        ],
    },
}


class FgosComparator:
    """Compares market-derived competencies with FGOS educational standards.

    Identifies curriculum gaps: what FGOS requires but market doesn't need,
    and what market demands but FGOS doesn't teach.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def compare_specialty(
        self,
        specialty: str,
        fgos_code: str,
    ) -> dict:
        """Compare market skills for a specialty against an FGOS standard.

        Args:
            specialty: Market specialty key (e.g. 'physics').
            fgos_code: FGOS specialty code (e.g. '03.03.02').

        Returns dict with:
        - specialty, fgos_code, fgos_name
        - market_skills: top 50 market-demanded skills
        - fgos_competencies: list of FGOS competency descriptions
        - market_coverage: % of FGOS competencies addressable by market skills
        - gaps: skills market wants but FGOS doesn't mention
        - recommendations: actionable curriculum update suggestions
        """
        fgos_entry = FGOS_TEMPLATES.get(fgos_code)
        if fgos_entry is None:
            return {
                "специальность": specialty,
                "код_фгос": fgos_code,
                "название_фгос": "неизвестно",
                "рыночные_навыки": [],
                "компетенции_фгос": [],
                "покрытие_рынком": 0.0,
                "разрывы": [],
                "рекомендации": [],
                "ошибка": f"Шаблон ФГОС не найден для кода: {fgos_code}",
                "создано": datetime.now(UTC).isoformat(),
            }

        freq_data = self._db.get_skill_frequencies(specialty, top_n=50)
        market_skills = [f.lemma for f in freq_data]

        fgos_competencies: list[str] = fgos_entry["компетенции"]

        # Compute coverage: for each FGOS competency, check if any market skill
        # appears as a lemma substring match in the competency text.
        covered = 0
        gap_skills: list[str] = []
        market_skill_set: set[str] = set(market_skills)

        for comp_text in fgos_competencies:
            comp_lower = comp_text.lower()
            matched = any(
                len(skill) >= 4 and skill in comp_lower
                for skill in market_skill_set
            )
            if matched:
                covered += 1

        total_fgos = len(fgos_competencies)
        coverage_pct = (covered / total_fgos * 100.0) if total_fgos > 0 else 0.0

        # Gaps: market skills not matched in any FGOS competency text
        all_fgos_text = " ".join(fgos_competencies).lower()
        for skill in market_skills:
            if len(skill) >= 4 and skill not in all_fgos_text:
                gap_skills.append(skill)

        recommendations = self._build_recommendations(
            gap_skills, freq_data, max_rec=10
        )

        return {
            "специальность": specialty,
            "код_фгос": fgos_code,
            "название_фгос": fgos_entry["название"],
            "рыночные_навыки": market_skills,
            "компетенции_фгос": fgos_competencies,
            "покрытие_рынком": round(coverage_pct, 1),
            "покрыто_компетенций": covered,
            "всего_компетенций_фгос": total_fgos,
            "разрывы": gap_skills,
            "рекомендации": recommendations,
            "создано": datetime.now(UTC).isoformat(),
        }

    def generate_recommendations(
        self,
        specialty: str,
        fgos_code: str,
        max_recommendations: int = 10,
    ) -> list[str]:
        """Generate actionable curriculum recommendations.

        Each recommendation: "Добавить в учебный план: [skill] (частота на рынке: N%)"
        """
        freq_data = self._db.get_skill_frequencies(specialty, top_n=200)
        comparison = self.compare_specialty(specialty, fgos_code)
        gap_skills: list[str] = comparison.get("разрывы", [])

        return self._build_recommendations(gap_skills, freq_data, max_recommendations)

    def curriculum_optimization_report(
        self,
        specialty: str,
        fgos_code: str,
    ) -> str:
        """Full curriculum optimization report combining market data and FGOS.

        Includes: market skill frequencies, FGOS coverage analysis,
        gap analysis, prioritized recommendations, simulation results.
        """
        comparison = self.compare_specialty(specialty, fgos_code)

        lines: list[str] = []
        lines.append(
            f"Отчёт оптимизации учебного плана: {specialty} ↔ {fgos_code}"
        )
        lines.append("=" * 60)
        lines.append("")

        fgos_name = comparison.get("название_фгос", "неизвестно")
        lines.append(f"Направление ФГОС: {fgos_code} — {fgos_name}")
        lines.append("")

        lines.append("--- Рыночные навыки (топ-20) ---")
        market_skills: list[str] = comparison.get("рыночные_навыки", [])
        for i, skill in enumerate(market_skills[:20], 1):
            lines.append(f"  {i:>2}. {skill}")
        lines.append("")

        lines.append("--- Покрытие ФГОС рыночными навыками ---")
        coverage = comparison.get("покрытие_рынком", 0.0)
        covered = comparison.get("покрыто_компетенций", 0)
        total = comparison.get("всего_компетенций_фгос", 0)
        lines.append(f"  Покрыто: {covered}/{total} компетенций ({coverage:.1f}%)")
        lines.append("")

        lines.append("--- Разрывы: навыки рынка вне ФГОС ---")
        gaps: list[str] = comparison.get("разрывы", [])
        if gaps:
            for i, gap in enumerate(gaps[:15], 1):
                lines.append(f"  {i:>2}. {gap}")
        else:
            lines.append("  Разрывы не обнаружены.")
        lines.append("")

        lines.append("--- Рекомендации по обновлению учебного плана ---")
        recs: list[str] = comparison.get("рекомендации", [])
        if recs:
            for i, rec in enumerate(recs, 1):
                lines.append(f"  {i:>2}. {rec}")
        else:
            lines.append("  Рекомендации не требуются.")
        lines.append("")

        lines.append(
            f"Отчёт создан: {comparison.get('создано', '')}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendations(
        gap_skills: list[str],
        freq_data: list,
        max_rec: int,
    ) -> list[str]:
        """Build prioritized curriculum recommendations from gap skills.

        Each recommendation includes the market frequency percentage.
        """
        from hh_competency.storage.models import SkillFrequency

        freq_map: dict[str, int] = {}
        max_freq = 1
        for f in freq_data:
            if isinstance(f, SkillFrequency):
                freq_map[f.lemma] = f.frequency
                if f.frequency > max_freq:
                    max_freq = f.frequency

        recommendations: list[str] = []
        for skill in gap_skills[:max_rec]:
            freq = freq_map.get(skill, 0)
            pct = (freq / max_freq * 100.0) if max_freq > 0 else 0.0
            recommendations.append(
                f"Добавить в учебный план: {skill} (частота на рынке: {pct:.0f}%)"
            )

        return recommendations
