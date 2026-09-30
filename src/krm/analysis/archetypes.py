"""ISCB-inspired role archetype classifier.

Classifies competency roles into six archetypes (Техник, Исследователь,
Методолог, Инженер, Ведущий специалист, Гибрид) based on the distribution
of their defining skills across four super-categories:
  - experimental  — lab/instrument/field/equipment skills
  - computational — programming, modelling, numerics
  - communication — data analysis + publications + project management
  - domain        — deep discipline-specific knowledge
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from krm.lib.skill_categories import classify_skill_to_category as _classify_skill_to_category


# ────────────────────────────────────────────────────────
# Archetype type definition
# ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoleArchetype:
    """Defines an ISCB-inspired competency role archetype.

    Each archetype describes what the role does, the typical career
    progression, and the competency areas that should be emphasised in
    training and evaluation.

    Attributes:
        name: Human-readable archetype name (Russian).
        description: Paragraph describing the role's focus and responsibilities.
        primary_super_category: Dominant super-category for this archetype
            ('experimental', 'computational', 'communication', 'domain'),
            or empty string if the archetype is multi-category.
        min_categories: Minimum number of significant super-categories
            (fraction > 15%) required for classification.
        max_super_frac: Upper bound for the primary super-category fraction
            (above this, the mono-category rules fire first).
        career_path: Typical career progression from entry to senior.
        competency_emphasis: Core competency areas for training and evaluation.
    """

    name: str
    description: str
    primary_super_category: str
    min_categories: int
    max_super_frac: float
    career_path: str
    competency_emphasis: list[str]


# Maps heuristic category names to one of four super-categories used
# by archetype classification.
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


# ────────────────────────────────────────────────────────
# ISCB-inspired role archetypes
# ────────────────────────────────────────────────────────

# Each archetype describes what the role does, the typical career
# progression, and the competency areas that should be emphasised
# in training and evaluation.

_ROLE_ARCHETYPES: list[RoleArchetype] = [
    RoleArchetype(
        name="Техник",
        description=(
            "Специалист по экспериментальной работе: владеет лабораторным "
            "оборудованием, методиками измерений, пробоподготовкой и полевыми "
            "методами. Основной фокус — точное и воспроизводимое выполнение "
            "экспериментальных протоколов."
        ),
        primary_super_category="experimental",
        min_categories=1,
        max_super_frac=0.5,
        career_path=(
            "Лаборант → Техник → Старший техник → "
            "Руководитель лабораторной группы → Начальник лаборатории"
        ),
        competency_emphasis=[
            "Владение оборудованием и приборами",
            "Соблюдение методик и протоколов",
            "Документирование экспериментальных данных",
            "Техника безопасности и нормативы",
            "Обслуживание и калибровка приборов",
        ],
    ),
    RoleArchetype(
        name="Исследователь",
        description=(
            "Классический учёный-исследователь: сочетает теоретические знания "
            "с экспериментальной проверкой гипотез. Балансирует между "
            "доменным знанием, вычислительными методами и экспериментальной "
            "работой."
        ),
        primary_super_category="",
        min_categories=3,
        max_super_frac=1.0,
        career_path="МНС → НС → СНС → ВНС → ГНС / Зав. лабораторией",
        competency_emphasis=[
            "Постановка научных задач и гипотез",
            "Дизайн экспериментов и исследований",
            "Анализ и интерпретация результатов",
            "Научные публикации и доклады",
            "Междисциплинарная коллаборация",
        ],
    ),
    RoleArchetype(
        name="Методолог",
        description=(
            "Специалист по методам анализа данных и научной коммуникации: "
            "статистическая обработка, визуализация, подготовка публикаций, "
            "методологическая поддержка исследовательских проектов. "
            "Высокая доля коммуникационных и аналитических навыков."
        ),
        primary_super_category="communication",
        min_categories=1,
        max_super_frac=0.35,
        career_path=(
            "Аналитик → Старший аналитик → Ведущий методолог → "
            "Руководитель аналитической группы → Директор по данным"
        ),
        competency_emphasis=[
            "Статистический анализ и визуализация",
            "Управление данными и базами данных",
            "Подготовка научных публикаций",
            "Грантовая поддержка и заявки",
            "Методологический аудит исследований",
        ],
    ),
    RoleArchetype(
        name="Инженер",
        description=(
            "Специалист по вычислительным методам и программированию: "
            "математическое моделирование, разработка ПО, HPC, машинное "
            "обучение. Основной фокус — создание и применение вычислительных "
            "инструментов для научных задач."
        ),
        primary_super_category="computational",
        min_categories=1,
        max_super_frac=0.5,
        career_path=(
            "Младший разработчик → Инженер-исследователь → "
            "Ведущий инженер → Руководитель IT/вычислительной группы → "
            "Технический директор (CTO)"
        ),
        competency_emphasis=[
            "Программирование и разработка ПО",
            "Математическое и численное моделирование",
            "Высокопроизводительные вычисления (HPC)",
            "Машинное обучение и AI",
            "Инженерная документация",
        ],
    ),
    RoleArchetype(
        name="Ведущий специалист",
        description=(
            "Опытный специалист с широким профилем: сочетает глубокие "
            "доменные знания с управленческими и коммуникационными навыками. "
            "Способен руководить междисциплинарными проектами, "
            "взаимодействовать с заказчиками и определять стратегию."
        ),
        primary_super_category="",
        min_categories=3,
        max_super_frac=1.0,
        career_path=(
            "Ведущий специалист → Руководитель направления → "
            "Научный руководитель → Директор института / "
            "Главный научный сотрудник"
        ),
        competency_emphasis=[
            "Стратегическое планирование исследований",
            "Управление проектами и командами",
            "Научная экспертиза и рецензирование",
            "Взаимодействие с промышленностью",
            "Наставничество и развитие кадров",
        ],
    ),
    RoleArchetype(
        name="Гибрид",
        description=(
            "Междисциплинарный специалист: владеет методами из двух и более "
            "различных доменных областей. Способен интегрировать подходы "
            "из разных научных дисциплин для решения комплексных задач."
        ),
        primary_super_category="",
        min_categories=2,
        max_super_frac=1.0,
        career_path=(
            "Специалист междисциплинарной группы → "
            "Координатор междисциплинарных проектов → "
            "Руководитель междисциплинарного центра → "
            "Директор по инновациям"
        ),
        competency_emphasis=[
            "Интеграция методов из разных дисциплин",
            "Системное мышление и синтез",
            "Коммуникация между научными областями",
            "Управление неопределённостью",
            "Трансфер технологий и знаний",
        ],
    ),
]

_ARCHETYPE_BY_NAME: dict[str, RoleArchetype] = {
    a.name: a for a in _ROLE_ARCHETYPES
}


# ────────────────────────────────────────────────────────
# Role input protocol
# ────────────────────────────────────────────────────────


class _HasDefiningSkills(Protocol):
    """Protocol for role-like objects accepted by classify_archetypes."""

    role_name: str
    defining_skills: list[tuple[str, float]]


# ────────────────────────────────────────────────────────
# Classification result
# ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchetypeClassification:
    """Result of classifying a single role into an archetype.

    Attributes:
        role_name: Name of the role being classified.
        archetype: The RoleArchetype assigned by the decision tree.
        super_fractions: Diagnostic dict of super-category fractions
            (experimental, computational, communication, domain).
        significant_categories: Number of categories above the 15% threshold.
    """

    role_name: str
    archetype: RoleArchetype
    super_fractions: dict[str, float]
    significant_categories: int


# ────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────


def classify_archetypes(
    roles: list[_HasDefiningSkills],
    discovered_axes: list[str],
) -> list[ArchetypeClassification]:
    """Classify each role into an ISCB-inspired archetype.

    Classification is data-driven: each role's ``defining_skills`` are
    mapped to heuristic categories (via :func:`_classify_skill_to_category`),
    then bucketed into four super-categories (experimental, computational,
    communication, domain) and classified by distribution shape.

    Decision tree (order matters — more specific rules first)::

        ┌─────────────────────┬──────────────────────────────────────┐
        │ Archetype            │ Heuristic                           │
        ├─────────────────────┼──────────────────────────────────────┤
        │ Техник               │ >50% skills in Experimental         │
        │ Инженер              │ >50% skills in Computational        │
        │ Исследователь        │ ≥3 significant categories,          │
        │                      │   communication < 30%               │
        │ Ведущий специалист   │ ≥3 significant categories,          │
        │                      │   communication ≥ 30%               │
        │ Методолог            │ ≥35% Communication                  │
        │ Гибрид               │ ≥2 significant categories           │
        │ (default Гибрид)     │ Unusual distribution                │
        └─────────────────────┴──────────────────────────────────────┘

    A category is "significant" if its fraction exceeds 15%.

    Args:
        roles: Discovered role profiles. Each must have ``role_name`` (str)
            and ``defining_skills`` (list of ``(skill_lemma, weight)`` tuples).
        discovered_axes: Names of statistically discovered competency axes
            (reserved for future data-driven tuning).

    Returns:
        One ``ArchetypeClassification`` per role, containing the assigned
        archetype, diagnostic fractions, and significance count.
    """
    results: list[ArchetypeClassification] = []

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

        fractions = {
            "experimental": round(exp_frac, 4),
            "computational": round(comp_frac, 4),
            "communication": round(comm_frac, 4),
            "domain": round(domain_frac, 4),
        }

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

        results.append(
            ArchetypeClassification(
                role_name=role.role_name,
                archetype=archetype_def,
                super_fractions=fractions,
                significant_categories=sig_cats,
            )
        )

    return results
