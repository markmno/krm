"""Role model builder: constructs formal competency-role model from market data.
allow: SIZE_OK — declarative data tables (RAN levels, archetype definitions,
super-category mapping) account for ~150 LOC; remaining ~260 LOC of logic
across 3 methods all serve the single responsibility "competency role model
building" with no separable sub-module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hh_competency.storage.models import CompetencyAxis, CompetencyMatrix, RoleArchetype

if TYPE_CHECKING:
    from hh_competency.storage.db import Database
    from hh_competency.storage.models import RoleProfile

# ── РАН 5-grade career levels ────────────────────────────────

_RAN_LEVELS: tuple[str, ...] = (
    "младший научный сотрудник",
    "научный сотрудник",
    "старший научный сотрудник",
    "ведущий научный сотрудник",
    "главный научный сотрудник",
)

_LEVEL_SCALING: dict[str, float] = {
    "младший научный сотрудник": 0.2,
    "научный сотрудник": 0.4,
    "старший научный сотрудник": 0.6,
    "ведущий научный сотрудник": 0.8,
    "главный научный сотрудник": 1.0,
}

_LEVEL_DESCRIPTIONS: dict[str, str] = {
    "младший научный сотрудник": (
        "Выполнение отдельных научных задач под руководством. "
        "Освоение базовых методов исследования, работа с оборудованием, "
        "сбор и первичная обработка данных."
    ),
    "научный сотрудник": (
        "Самостоятельное проведение исследований в рамках утверждённых методик. "
        "Анализ и интерпретация результатов, подготовка научных публикаций, "
        "участие в планировании экспериментов."
    ),
    "старший научный сотрудник": (
        "Руководство исследовательскими проектами и научными группами. "
        "Разработка методик, постановка научных задач, экспертиза результатов, "
        "подготовка заявок на финансирование."
    ),
    "ведущий научный сотрудник": (
        "Определение научных направлений и стратегии исследований. "
        "Координация междисциплинарных проектов, научное руководство, "
        "взаимодействие с промышленными партнёрами."
    ),
    "главный научный сотрудник": (
        "Стратегическое научное руководство, формирование научной школы. "
        "Экспертиза национального и международного уровня, "
        "определение приоритетов развития научной области."
    ),
}

# ── Super-category mapping for archetype classification ─────

# Maps heuristic category names from axis_discovery._classify_skill_to_category()
# into the four super-categories used by archetype classification:
#   experimental — lab/instrument/field/equipment skills
#   computational — programming, modelling, numerics
#   communication — data analysis + publications + project management
#   domain — deep discipline-specific knowledge

_SUPER_CATEGORIES: dict[str, str] = {
    "Экспериментальные методы и оборудование": "experimental",
    "Биологические и полевые методы": "experimental",
    "Химический синтез и технология": "experimental",
    "Программирование и IT": "computational",
    "Вычислительное моделирование": "computational",
    "Анализ данных": "communication",
    "Научная коммуникация и публикации": "communication",
    "Управление проектами и нормативы": "communication",
    "Фундаментальная физика и материаловедение": "domain",
}

# ── ISCB-inspired role archetypes ────────────────────────────

# Each archetype describes: what the role does, the typical career
# progression, and the competency areas that should be emphasised
# in training and evaluation.

_ROLE_ARCHETYPES: list[dict] = [
    {
        "name": "Техник",
        "description": (
            "Специалист по экспериментальной работе: владеет лабораторным "
            "оборудованием, методиками измерений, пробоподготовкой и полевыми "
            "методами. Основной фокус — точное и воспроизводимое выполнение "
            "экспериментальных протоколов."
        ),
        "career_path": (
            "Лаборант → Техник → Старший техник → "
            "Руководитель лабораторной группы → Начальник лаборатории"
        ),
        "competency_emphasis": [
            "Владение оборудованием и приборами",
            "Соблюдение методик и протоколов",
            "Документирование экспериментальных данных",
            "Техника безопасности и нормативы",
            "Обслуживание и калибровка приборов",
        ],
    },
    {
        "name": "Исследователь",
        "description": (
            "Классический учёный-исследователь: сочетает теоретические знания "
            "с экспериментальной проверкой гипотез. Балансирует между "
            "доменным знанием, вычислительными методами и экспериментальной "
            "работой."
        ),
        "career_path": (
            "МНС → НС → СНС → ВНС → ГНС / Зав. лабораторией"
        ),
        "competency_emphasis": [
            "Постановка научных задач и гипотез",
            "Дизайн экспериментов и исследований",
            "Анализ и интерпретация результатов",
            "Научные публикации и доклады",
            "Междисциплинарная коллаборация",
        ],
    },
    {
        "name": "Методолог",
        "description": (
            "Специалист по методам анализа данных и научной коммуникации: "
            "статистическая обработка, визуализация, подготовка публикаций, "
            "методологическая поддержка исследовательских проектов. "
            "Высокая доля коммуникационных и аналитических навыков."
        ),
        "career_path": (
            "Аналитик → Старший аналитик → Ведущий методолог → "
            "Руководитель аналитической группы → Директор по данным"
        ),
        "competency_emphasis": [
            "Статистический анализ и визуализация",
            "Управление данными и базами данных",
            "Подготовка научных публикаций",
            "Грантовая поддержка и заявки",
            "Методологический аудит исследований",
        ],
    },
    {
        "name": "Инженер",
        "description": (
            "Специалист по вычислительным методам и программированию: "
            "математическое моделирование, разработка ПО, HPC, машинное "
            "обучение. Основной фокус — создание и применение вычислительных "
            "инструментов для научных задач."
        ),
        "career_path": (
            "Младший разработчик → Инженер-исследователь → "
            "Ведущий инженер → Руководитель IT/вычислительной группы → "
            "Технический директор (CTO)"
        ),
        "competency_emphasis": [
            "Программирование и разработка ПО",
            "Математическое и численное моделирование",
            "Высокопроизводительные вычисления (HPC)",
            "Машинное обучение и AI",
            "Инженерная документация",
        ],
    },
    {
        "name": "Ведущий специалист",
        "description": (
            "Опытный специалист с широким профилем: сочетает глубокие "
            "доменные знания с управленческими и коммуникационными навыками. "
            "Способен руководить междисциплинарными проектами, "
            "взаимодействовать с заказчиками и определять стратегию."
        ),
        "career_path": (
            "Ведущий специалист → Руководитель направления → "
            "Научный руководитель → Директор института / "
            "Главный научный сотрудник"
        ),
        "competency_emphasis": [
            "Стратегическое планирование исследований",
            "Управление проектами и командами",
            "Научная экспертиза и рецензирование",
            "Взаимодействие с промышленностью",
            "Наставничество и развитие кадров",
        ],
    },
    {
        "name": "Гибрид",
        "description": (
            "Междисциплинарный специалист: владеет методами из двух и более "
            "различных доменных областей. Способен интегрировать подходы "
            "из разных научных дисциплин для решения комплексных задач."
        ),
        "career_path": (
            "Специалист междисциплинарной группы → "
            "Координатор междисциплинарных проектов → "
            "Руководитель междисциплинарного центра → "
            "Директор по инновациям"
        ),
        "competency_emphasis": [
            "Интеграция методов из разных дисциплин",
            "Системное мышление и синтез",
            "Коммуникация между научными областями",
            "Управление неопределённостью",
            "Трансфер технологий и знаний",
        ],
    },
]

_ARCHETYPE_BY_NAME: dict[str, dict] = {a["name"]: a for a in _ROLE_ARCHETYPES}


# ── Public API ───────────────────────────────────────────────


class CompetencyRoleBuilder:
    """Constructs formal competency-role model from discovered market data.

    Takes clustered roles (from SkillClusterer) and builds the multi-axis
    competency matrix: role level × competency axis → required proficiency.

    Architecture:
    - РАН 5-grade system (мнс→гнс) mapped to vacancy experience levels
    - Competency axes discovered statistically via AxisDiscoveryEngine
    - ISCB-inspired role archetypes classified from skill distributions
    - Proficiency from market frequency
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── Competency matrix (data-driven axes) ─────────────────

    def build_competency_matrix(
        self,
        specialty: str,
        roles: list[RoleProfile] | None = None,
    ) -> CompetencyMatrix:
        """Build full competency matrix using data-driven axis discovery.

        Uses AxisDiscoveryEngine to statistically discover competency axes
        from skill co-occurrence patterns, then computes РАН 5-grade
        proficiency levels for each axis at each rank.

        Args:
            specialty: Specialty identifier (e.g. 'physics').
            roles: Optional pre-discovered role profiles. Kept for backward
                   compatibility; axis discovery is now purely data-driven.

        Returns:
            CompetencyMatrix with statistically discovered axes and a
            level × axis proficiency matrix.
        """
        from hh_competency.analysis.axis_discovery import AxisDiscoveryEngine

        # Fetch skill frequencies and vacancy count for axis discovery
        freq_data = self._db.get_skill_frequencies(specialty, top_n=500)
        n_vacancies = self._db.get_vacancy_count(specialty)

        # Discover axes statistically from co-occurrence data
        engine = AxisDiscoveryEngine(self._db)
        discovery = engine.discover(
            specialty=specialty,
            skills=freq_data,
            n_vacancies=n_vacancies,
        )

        # Build CompetencyAxis objects from discovered axis definitions
        axes: list[CompetencyAxis] = []
        for axis_def in discovery.discovered_axes:
            axes.append(
                CompetencyAxis(
                    name=axis_def.name,
                    typical_skills=axis_def.defining,
                )
            )

        # Fallback: if discovery returns no axes (too few skills), produce
        # a minimal single-axis matrix so downstream code gets a valid result.
        if not axes:
            skill_lemmas = [f.lemma for f in freq_data[:5]] if freq_data else ["общие навыки"]
            axes.append(
                CompetencyAxis(
                    name="Общие компетенции",
                    typical_skills=skill_lemmas,
                )
            )

        # Compute proficiency matrix: axis base from market frequency,
        # scaled by РАН level
        lemma_to_freq: dict[str, int] = {f.lemma: f.frequency for f in freq_data}
        max_freq = max(lemma_to_freq.values()) if lemma_to_freq else 1

        axis_base: list[float] = []
        for axis in axes:
            freqs = [lemma_to_freq.get(skill, 0) for skill in axis.typical_skills]
            avg_freq = sum(freqs) / len(freqs) if freqs else 0.0
            axis_base.append(min(5.0, (avg_freq / max_freq) * 5.0))

        matrix: list[list[int]] = []
        for level in _RAN_LEVELS:
            scale = _LEVEL_SCALING[level]
            row = [max(1, int(round(b * scale))) for b in axis_base]
            matrix.append(row)

        return CompetencyMatrix(
            specialty=specialty,
            axes=axes,
            role_levels=list(_RAN_LEVELS),
            matrix=matrix,
        )

    # ── Full role model (matrix + profiles + archetypes) ─────

    def build_role_model(
        self,
        specialty: str,
        output_format: str = "dict",
    ) -> dict:
        """Build complete competency-role model with archetype classification.

        Returns dict with:
        - specialty name
        - role_profiles (list of role defs with defining/supporting skills)
        - competency_matrix (axes × levels, data-driven)
        - level_descriptions (мнс→гнс behavior expectations)
        - archetypes (ISCB-inspired archetype definitions)
        - role_archetypes (mapping: role_name → archetype_name)
        - generated_at timestamp
        """
        from hh_competency.analysis.clustering import SkillClusterer

        clusterer = SkillClusterer(self._db)
        roles = clusterer.discover_roles(specialty)
        matrix = self.build_competency_matrix(specialty, roles)

        # Classify roles into ISCB-inspired archetypes
        discovered_axis_names = [ax.name for ax in matrix.axes]
        role_archetypes = self.classify_archetypes(roles, discovered_axis_names)

        role_profiles: list[dict] = []
        for role in roles:
            role_profiles.append(
                {
                    "название_роли": role.role_name,
                    "определяющие_навыки": [
                        {"навык": skill, "вес": round(weight, 4)}
                        for skill, weight in role.defining_skills
                    ],
                    "поддерживающие_навыки": [
                        {"навык": skill, "вес": round(weight, 4)}
                        for skill, weight in role.supporting_skills
                    ],
                }
            )

        # Deduplicated archetype summaries for archetypes present in this specialty
        seen_names: set[str] = set()
        archetype_summaries: list[dict] = []
        role_mapping: dict[str, str] = {}
        for ra in role_archetypes:
            role_mapping[ra.role_name] = ra.archetype_name
            if ra.archetype_name not in seen_names:
                seen_names.add(ra.archetype_name)
                a = _ARCHETYPE_BY_NAME.get(ra.archetype_name)
                if a:
                    archetype_summaries.append({
                        "название": a["name"],
                        "описание": a["description"],
                        "карьерный_путь": a["career_path"],
                        "компетентностный_акцент": a["competency_emphasis"],
                    })

        return {
            "специальность": specialty,
            "ролевые_профили": role_profiles,
            "компетентностная_матрица": {
                "специальность": matrix.specialty,
                "оси": [
                    {"название": a.name, "типовые_навыки": a.typical_skills}
                    for a in matrix.axes
                ],
                "уровни": matrix.role_levels,
                "матрица_владения": matrix.matrix,
            },
            "описания_уровней": _LEVEL_DESCRIPTIONS,
            "архетипы": archetype_summaries,
            "роли_архетипы": role_mapping,
            "создано": datetime.now(UTC).isoformat(),
        }

    # ── Archetype classification ─────────────────────────────

    def classify_archetypes(
        self,
        roles: list[RoleProfile],
        discovered_axes: list[str],
    ) -> list[RoleArchetype]:
        """Classify each role into an ISCB-inspired archetype.

        Classification is data-driven: each role's defining_skills are
        mapped to heuristic categories (via axis_discovery), then bucketed
        into four super-categories (experimental, computational,
        communication, domain) and classified by distribution shape:

        ┌─────────────────┬──────────────────────────────────────┐
        │ Archetype        │ Heuristic                           │
        ├─────────────────┼──────────────────────────────────────┤
        │ Техник           │ >50% skills in Experimental         │
        │ Инженер          │ >50% skills in Computational        │
        │ Методолог        │ ≥30% Communication + domain/exp     │
        │ Исследователь    │ ≥3 significant categories, balanced │
        │ Ведущий специалист│ ≥3 significant, ≥25% Communication │
        │ Гибрид           │ ≥2 significant domain categories    │
        └─────────────────┴──────────────────────────────────────┘

        Args:
            roles: Discovered role profiles from SkillClusterer.
            discovered_axes: Names of statistically discovered axes
                             (reserved for future data-driven tuning).

        Returns:
            One RoleArchetype per role, with name, description,
            career path, and competency emphasis.
        """
        from hh_competency.analysis.axis_discovery import _classify_skill_to_category

        results: list[RoleArchetype] = []
        for role in roles:
            # Classify every defining skill into a heuristic category
            skill_categories: list[str | None] = [
                _classify_skill_to_category(skill)
                for skill, _weight in role.defining_skills
            ]

            # Map to super-categories and compute fractions
            super_counts: dict[str, float] = {
                "experimental": 0.0,
                "computational": 0.0,
                "communication": 0.0,
                "domain": 0.0,
            }
            n_classified = 0
            for cat in skill_categories:
                if cat and cat in _SUPER_CATEGORIES:
                    super_counts[_SUPER_CATEGORIES[cat]] += 1.0
                    n_classified += 1

            total = max(n_classified, 1)
            exp_frac = super_counts["experimental"] / total
            comp_frac = super_counts["computational"] / total
            comm_frac = super_counts["communication"] / total
            domain_frac = super_counts["domain"] / total

            # Count super-categories with significant presence (>15%)
            sig_threshold = 0.15
            sig_cats = sum(
                1
                for v in [exp_frac, comp_frac, comm_frac, domain_frac]
                if v > sig_threshold
            )

            # Determine archetype from distribution shape.
            # Order matters: more specific rules first, then broader.
            archetype_name: str
            if exp_frac > 0.5:
                archetype_name = "Техник"
            elif comp_frac > 0.5:
                archetype_name = "Инженер"
            elif sig_cats >= 3 and comm_frac < 0.3:
                # Balanced domain + experimental + computational, low communication
                archetype_name = "Исследователь"
            elif sig_cats >= 3 and comm_frac >= 0.3:
                # Mix of all categories including significant communication
                archetype_name = "Ведущий специалист"
            elif comm_frac >= 0.35:
                # Communication-dominated, not broadly balanced
                archetype_name = "Методолог"
            elif sig_cats >= 2:
                archetype_name = "Гибрид"
            else:
                archetype_name = "Гибрид"  # Unusual distribution, default to hybrid

            archetype_def = _ARCHETYPE_BY_NAME.get(
                archetype_name, _ARCHETYPE_BY_NAME["Гибрид"]
            )
            results.append(RoleArchetype(
                role_name=role.role_name,
                archetype_name=archetype_name,
                description=archetype_def["description"],
                career_path=archetype_def["career_path"],
                competency_emphasis=archetype_def["competency_emphasis"],
            ))

        return results
