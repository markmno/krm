"""Profstandart mapper: aligns discovered market roles with Russian profstandart framework.

Maps competency-role profiles to the Russian professional standards framework
(профессиональные стандарты) with 9 qualification levels and universal competencies.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hh_competency.storage.db import Database
    from hh_competency.storage.models import RoleProfile

# -- Universal competencies per Russian profstandart methodology --
UNIVERSAL_COMPETENCIES: tuple[str, ...] = (
    "системное мышление",
    "междисциплинарность",
    "управление проектами",
    "командная работа",
    "коммуникация",
    "самоорганизация",
    "цифровая грамотность",
    "информационная безопасность",
    "правовая грамотность",
    "инновационная деятельность",
    "экологическое мышление",
    "бережливое производство",
    "гражданская позиция",
)

# -- Profstandart 9-grade qualification levels --
PROFESSIONAL_LEVELS: dict[int, str] = {
    1: "Исполнительская деятельность под руководством",
    2: "Деятельность под руководством с элементами самостоятельности",
    3: "Деятельность под руководством с проявлением самостоятельности",
    4: "Деятельность, предполагающая решение задач и управление сотрудниками",
    5: "Самостоятельная профессиональная деятельность",
    6: "Управление крупной организацией или проектами",
    7: "Управление сложными видами деятельности",
    8: "Управление научной или проектной деятельностью",
    9: "Управление сложными социальными и экономическими системами",
}

# -- РАН level → profstandart level mapping --
RAN_TO_PROFSTANDART: dict[str, int] = {
    "младший научный сотрудник": 5,
    "научный сотрудник": 6,
    "старший научный сотрудник": 7,
    "ведущий научный сотрудник": 8,
    "главный научный сотрудник": 9,
}

_RAN_LEVELS: tuple[str, ...] = (
    "младший научный сотрудник",
    "научный сотрудник",
    "старший научный сотрудник",
    "ведущий научный сотрудник",
    "главный научный сотрудник",
)


class ProfstandartMapper:
    """Maps discovered competency roles to Russian profstandart framework.

    Aligns market-derived competency clusters with the 9-level Russian
    professional standards system and universal competency taxonomy.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def map_roles_to_profstandart(
        self,
        specialty: str,
        roles: list[RoleProfile] | None = None,
    ) -> dict:
        """Map each discovered role to closest profstandart match.

        Returns dict with:
        - specialty, roles (list of role→profstandart mappings)
        - each role mapping: role_name, profstandart_level, level_description,
          matched_competencies, universal_competencies
        """
        resolved_roles = roles
        if resolved_roles is None:
            from hh_competency.analysis.clustering import SkillClusterer

            clusterer = SkillClusterer(self._db)
            resolved_roles = clusterer.discover_roles(specialty)

        if not resolved_roles:
            return {
                "специальность": specialty,
                "роли": [],
                "создано": datetime.now(UTC).isoformat(),
            }

        # Rank roles by average skill frequency (proxy for specialization level)
        freq_data = self._db.get_skill_frequencies(specialty, top_n=500)
        lemma_to_freq: dict[str, int] = {f.lemma: f.frequency for f in freq_data}

        def _avg_freq(role: RoleProfile) -> float:
            freqs = [
                lemma_to_freq.get(skill, 0)
                for skill, _weight in role.defining_skills
            ]
            return sum(freqs) / len(freqs) if freqs else 0.0

        sorted_roles = sorted(resolved_roles, key=_avg_freq)

        n_levels = len(_RAN_LEVELS)
        role_mappings: list[dict] = []
        for i, role in enumerate(sorted_roles):
            level_index = min(i, n_levels - 1)
            ran_level = _RAN_LEVELS[level_index]
            ps_level = RAN_TO_PROFSTANDART[ran_level]

            matched = [
                {"навык": skill, "вес": round(weight, 4)}
                for skill, weight in role.defining_skills
            ]

            role_mappings.append(
                {
                    "название_роли": role.role_name,
                    "уровень_ран": ran_level,
                    "уровень_профстандарта": ps_level,
                    "описание_уровня": PROFESSIONAL_LEVELS[ps_level],
                    "сопоставленные_компетенции": matched,
                    "универсальные_компетенции": list(UNIVERSAL_COMPETENCIES),
                    "определяющие_навыки": [
                        skill for skill, _weight in role.defining_skills
                    ],
                    "поддерживающие_навыки": [
                        skill for skill, _weight in role.supporting_skills
                    ],
                }
            )

        return {
            "специальность": specialty,
            "роли": role_mappings,
            "создано": datetime.now(UTC).isoformat(),
        }

    def generate_profstandart_report(self, specialty: str) -> str:
        """Generate human-readable profstandart alignment report."""
        mapping = self.map_roles_to_profstandart(specialty)

        lines: list[str] = []
        lines.append(f"Профстандарт-отчёт для специальности: {specialty}")
        lines.append("=" * 60)
        lines.append("")

        if not mapping["роли"]:
            lines.append("Роли не обнаружены.")
            return "\n".join(lines)

        for role_map in mapping["роли"]:
            lines.append(f"Роль: {role_map['название_роли']}")
            lines.append(f"  Уровень РАН: {role_map['уровень_ран']}")
            lines.append(
                f"  Уровень профстандарта: {role_map['уровень_профстандарта']}"
            )
            lines.append(f"  Описание: {role_map['описание_уровня']}")
            lines.append("  Определяющие навыки:")
            for skill in role_map["определяющие_навыки"]:
                lines.append(f"    - {skill}")
            lines.append("  Универсальные компетенции:")
            for comp in role_map["универсальные_компетенции"][:5]:
                lines.append(f"    - {comp}")
            if len(role_map["универсальные_компетенции"]) > 5:
                lines.append(
                    f"    ... и ещё {len(role_map['универсальные_компетенции']) - 5}"
                )
            lines.append("")

        return "\n".join(lines)

    def compare_with_fgos(self, specialty: str, fgos_code: str) -> dict:
        """Compare discovered competencies against FGOS standard.

        Creates a FgosComparator and delegates the comparison.
        fgos_code e.g. '03.03.02' for Физика.
        """
        from hh_competency.methodology.fgos import FgosComparator

        comparator = FgosComparator(self._db)
        return comparator.compare_specialty(specialty, fgos_code)
