"""Generate the ideal competency-role model report (эталон).

Produces a self-contained HTML report — the reference for how the KRM
pipeline's final output should look — with all charts as interactive Plotly
divs (no PNG files). All data is illustrative mock data (not real pipeline
output), designed to show the ideal end-state.

The report renders a set of biology sub-specialty roles, each defined by a
DOMINANT PAIR of competences (its signature). The career trajectory (Junior →
C-level) is monotonic: every competence only grows as seniority increases; the
dominant pair stays ahead of the rest at every level and never declines at
C-level.

Usage:
    python -m scripts.generate_etalon
"""

from __future__ import annotations

import html
import math
from typing import Any, NotRequired, TypedDict

from krm.config import Config
from krm.visualization import (
    CareerLevel,
    render_comparison_html,
    render_leadership_gap_html,
    render_role_experience_html,
    render_role_skills_html,
    render_role_soft_spider_html,
    render_role_spider_html,
    render_summary_html,
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


class MockRole(TypedDict):
    role_id: int
    label: str
    top_titles: list[str]
    member_count: int
    skill_count: int
    dominant: list[str]
    junior: dict[str, float]
    clevel: dict[str, float]
    skills: dict[str, list[str]]
    example_vacancy: NotRequired[dict[str, Any]]


AXIS_IDS = [
    "domain_knowledge", "experimental", "data_analysis", "computational",
    "professional_texts", "t_profile", "management",
]

CAREER_LEVELS = [
    "Младший специалист",
    "Специалист",
    "Старший специалист",
    "Ведущий специалист",
    "Главный специалист",
    "Руководитель направления",
    "Топ-менеджмент (C-level)",
]

# Each role: a dominant pair + junior and C-level profiles. The trajectory is a
# monotonic interpolation junior → C-level (every axis grows, none declines).
ROLES: list[MockRole] = [
    {
        "role_id": 0,
        "label": "Биолог-экспериментатор",
        "top_titles": ["биолог-экспериментатор", "молекулярный биолог",
                       "научный сотрудник лаборатории"],
        "member_count": 284,
        "skill_count": 38,
        "dominant": ["experimental", "data_analysis"],
        "junior": {"experimental": 1.6, "data_analysis": 1.5, "domain_knowledge": 1.3,
                   "computational": 1.2, "professional_texts": 1.2, "management": 1.1,
                   "t_profile": 1.2},
        "clevel": {"experimental": 4.9, "data_analysis": 4.6, "domain_knowledge": 3.4,
                   "professional_texts": 3.2, "computational": 3.0, "management": 2.6,
                   "t_profile": 3.0},
        "skills": {
            "experimental": ["ПЦР", "культивирование клеток", "вестерн-блоттинг", "электрофорез"],
            "data_analysis": ["статистическая обработка", "анализ экспрессии генов",
                              "обработка экспериментальных данных",
                              "планирование эксперимента (DoE)"],
            "domain_knowledge": ["молекулярная биология", "клеточная биология",
                                 "биохимия", "генетика"],
            "professional_texts": ["научные публикации", "обзор литературы",
                                   "подготовка протоколов", "оформление методик"],
            "management": ["ведение лабораторного журнала", "планирование эксперимента",
                           "учёт реактивов", "работа с поставщиками"],
            "computational": ["R", "ImageJ", "базы данных последовательностей", "Python"],
            "t_profile": ["междисциплинарная эрудиция", "системное мышление",
                          "смежные естественные науки", "научная широта кругозора"],
        },
    },
    {
        "role_id": 1,
        "label": "Биоинформатик",
        "top_titles": ["биоинформатик", "биоинформатик / data-engineer",
                       "вычислительный биолог"],
        "member_count": 240,
        "skill_count": 34,
        "dominant": ["domain_knowledge", "computational"],
        "junior": {"domain_knowledge": 1.6, "computational": 1.5, "data_analysis": 1.3,
                   "professional_texts": 1.2, "experimental": 1.1, "management": 1.1,
                   "t_profile": 1.2},
        "clevel": {"domain_knowledge": 5.0, "computational": 4.7, "data_analysis": 3.3,
                   "professional_texts": 3.1, "management": 2.7, "experimental": 2.2,
                   "t_profile": 3.0},
        "skills": {
            "domain_knowledge": ["геномика", "молекулярная биология", "биохимия",
                                 "транскриптомика"],
            "computational": ["Python", "Snakemake", "выравнивание последовательностей",
                              "Bioconductor"],
            "data_analysis": ["статистический анализ", "машинное обучение",
                              "анализ RNA-seq", "дифференциальная экспрессия"],
            "professional_texts": ["научные публикации", "аннотация геномов",
                                   "обзор литературы", "метаанализ"],
            "management": ["ведение проектов", "документирование конвейеров",
                           "командная работа", "рецензирование кода"],
            "experimental": ["подготовка образцов", "секвенирование", "ПЦР", "выделение ДНК/РНК"],
            "t_profile": ["междисциплинарная эрудиция", "системное мышление",
                          "смежные STEM-области", "научная широта кругозора"],
        },
    },
    {
        "role_id": 2,
        "label": "Менеджер лаборатории",
        "top_titles": ["менеджер лаборатории", "менеджер научных проектов",
                       "координатор лаборатории"],
        "member_count": 218,
        "skill_count": 28,
        "dominant": ["domain_knowledge", "management"],
        "junior": {"management": 1.6, "domain_knowledge": 1.5, "professional_texts": 1.3,
                   "experimental": 1.2, "data_analysis": 1.2, "computational": 1.1,
                   "t_profile": 1.2},
        "clevel": {"management": 5.0, "domain_knowledge": 4.5, "professional_texts": 3.3,
                   "data_analysis": 2.6, "experimental": 2.2, "computational": 2.0,
                   "t_profile": 3.0},
        "skills": {
            "management": ["управление проектами", "планирование бюджета",
                           "координация команды", "управление закупками"],
            "domain_knowledge": ["биология лабораторных методов", "стандарты лаборатории",
                                 "методики исследований", "регламенты лаборатории"],
            "professional_texts": ["техническая документация", "отчётность",
                                   "взаимодействие с заказчиком", "подготовка договоров"],
            "data_analysis": ["контроль качества данных", "Excel", "метрики процессов",
                              "отчётность по KPI"],
            "experimental": ["понимание лабораторных методик", "эксплуатация оборудования",
                             "техника безопасности", "контроль соблюдения методик"],
            "computational": ["системы учёта", "ERP", "базы данных", "CRM-системы"],
            "t_profile": ["междисциплинарная эрудиция", "управленческая широта",
                          "смежные функциональные области", "системное мышление"],
        },
    },
    {
        "role_id": 3,
        "label": "Биотехнолог",
        "top_titles": ["биотехнолог", "инженер-биотехнолог", "инженер-исследователь"],
        "member_count": 310,
        "skill_count": 32,
        "dominant": ["experimental", "computational"],
        "junior": {"experimental": 1.6, "computational": 1.5, "domain_knowledge": 1.3,
                   "data_analysis": 1.2, "professional_texts": 1.1, "management": 1.1,
                   "t_profile": 1.2},
        "clevel": {"experimental": 4.8, "computational": 4.6, "domain_knowledge": 3.3,
                   "data_analysis": 3.0, "professional_texts": 2.6, "management": 2.6,
                   "t_profile": 3.0},
        "skills": {
            "experimental": ["биореакторы", "ферментация",
                             "культивирование микроорганизмов", "хроматография"],
            "computational": ["моделирование биопроцессов", "Python", "SCADA-системы",
                              "MATLAB"],
            "domain_knowledge": ["биотехнология", "микробиология", "биохимия",
                                 "генная инженерия"],
            "data_analysis": ["анализ биопроцессов", "статистика",
                              "мониторинг параметров", "оптимизация процессов"],
            "professional_texts": ["техническая документация", "регламенты",
                                   "отчёты", "стандарты GMP"],
            "management": ["планирование производства", "работа с поставщиками",
                           "ведение задач", "ведение регламентов"],
            "t_profile": ["междисциплинарная эрудиция", "инженерная широта",
                          "смежные технологии", "системное мышление"],
        },
    },
    {
        "role_id": 4,
        "label": "Биолог-исследователь",
        "top_titles": ["биолог-исследователь", "научный сотрудник", "аналитик научной литературы"],
        "member_count": 265,
        "skill_count": 30,
        "dominant": ["professional_texts", "experimental"],
        "junior": {"experimental": 1.5, "professional_texts": 1.5, "domain_knowledge": 1.3,
                   "data_analysis": 1.2, "computational": 1.1, "management": 1.1,
                   "t_profile": 1.2},
        "clevel": {"professional_texts": 4.8, "experimental": 4.6, "domain_knowledge": 3.4,
                   "data_analysis": 2.9, "management": 2.7, "computational": 2.3,
                   "t_profile": 3.0},
        "skills": {
            "professional_texts": ["научные публикации", "патентный поиск",
                                   "рецензирование", "написание обзоров"],
            "experimental": ["постановка экспериментов", "работа с образцами",
                             "микроскопия", "гистология"],
            "domain_knowledge": ["биология развития", "методология исследований",
                                 "систематика", "эволюционная биология"],
            "data_analysis": ["метаанализ", "обработка литературных данных",
                              "статистическая обработка", "библиометрия"],
            "management": ["координация публикаций", "ведение библиографии",
                           "работа с редакциями", "организация семинаров"],
            "computational": ["LaTeX", "Zotero", "базы публикаций", "EndNote"],
            "t_profile": ["междисциплинарная эрудиция", "научная широта кругозора",
                          "смежные дисциплины", "системное мышление"],
        },
    },
    {
        "role_id": 5,
        "label": "Биолог-аналитик данных",
        "top_titles": ["биолог-аналитик данных", "биостатистик", "data scientist"],
        "member_count": 265,
        "skill_count": 24,
        "dominant": ["data_analysis", "computational"],
        "junior": {"data_analysis": 1.6, "computational": 1.5, "domain_knowledge": 1.3,
                   "professional_texts": 1.2, "experimental": 1.1, "management": 1.1,
                   "t_profile": 1.2},
        "clevel": {"data_analysis": 5.0, "computational": 4.6, "domain_knowledge": 3.2,
                   "professional_texts": 2.9, "management": 2.7, "experimental": 2.0,
                   "t_profile": 3.0},
        "skills": {
            "data_analysis": ["машинное обучение", "статистическое моделирование",
                              "анализ больших данных", "байесовская статистика"],
            "computational": ["Python", "SQL", "R", "pandas/scikit-learn"],
            "domain_knowledge": ["биостатистика", "эпидемиология", "модели данных",
                                 "клинические исследования"],
            "professional_texts": ["отчётность", "визуализация результатов",
                                   "документация", "построение дашбордов"],
            "management": ["постановка аналитических задач", "работа с заказчиком",
                           "планирование", "управление данными"],
            "experimental": ["понимание биологических методик", "сбор данных",
                             "работа с образцами", "стандартизация проб"],
            "t_profile": ["междисциплинарная эрудиция", "смежные STEM-области",
                          "системное мышление", "научная широта кругозора"],
        },
    },
]

KPIS = [
    ("8 439", "Вакансий собрано"),
    ("6", "Ролей обнаружено"),
    ("1 739", "STEM-вакансий"),
    ("186", "Навыков извлечено"),
    ("91%", "Покрытие модели"),
]

METRICS = [
    ("Классификация STEM/IT (F1)", "0.94", "≥ 0.92"),
    ("Кластеризация ролей (Silhouette)", "0.47", "> 0.40"),
    ("Кластеризация (Davies–Bouldin)", "1.12", "ниже — лучше"),
    ("Стабильность ролей (bootstrap ARI)", "0.78", "> 0.70"),
    ("Извлечение навыков (Precision@10)", "0.88", "≥ 0.85"),
    ("Маппинг компетенций (Spearman ρ)", "0.81", "≥ 0.75"),
    ("Покрытие модели", "91%", "≥ 85%"),
]

# How each metric is computed and what it proves.
METRIC_EXPLANATIONS = [
    ("F1 (STEM/IT)",
     "Гармоническое среднее точности и полноты классификатора на размеченной выборке вакансий.",
     "Насколько надёжно пайплайн отделяет исследовательские вакансии от IT и непрофильных."),
    ("Silhouette",
     "Для каждой вакансии — насколько она ближе к своей роли, чем к соседним (шкала −1…+1).",
     "Роли — это плотные и хорошо разделённые кластеры, а не случайное разбиение."),
    ("Davies–Bouldin",
     "Среднее отношение внутрикластерного разброса к расстоянию между кластерами.",
     "Чем ниже — тем компактнее роли и тем чётче границы между ними."),
    ("Bootstrap ARI",
     "Насколько кластеризация воспроизводится на случайных 80% подвыборках (adjusted Rand index).",
     "Роли устойчивы: не распадаются и не сливаются при малом изменении данных."),
    ("Precision@10",
     "Доля релевантных навыков среди топ-10 извлечённых для роли (по экспертной разметке).",
     "Извлечённые навыки действительно описывают работу в этой роли."),
    ("Spearman ρ",
     "Ранговая корреляция между оценками компетенций модели и оценками экспертов.",
     "Модель ранжирует компетенции роли так же, как это делает эксперт."),
    ("Покрытие модели",
     "Доля STEM-вакансий, для которых построена полная модель (все 7 компетенций).",
     "Модель применима к значительной части рынка, а не к узкой выборке."),
]

# ---------------------------------------------------------------------------
# Technology readiness levels (УГТ/TRL) — borrowed from ГОСТ Р 71726-2024
# ---------------------------------------------------------------------------

# Four contiguous stage bands (cleaned up from the source's 0-3/3-4/5-6/7-9
# which overlapped at 3 and gapped at 4/6). Each band maps to a role family.
TRL_STAGES = [
    {"id": 0, "trl": "0–3", "name": "Фундаментальные исследования",
     "family": "Исследователь", "w": 3},
    {"id": 1, "trl": "4–5", "name": "Прикладные исследования",
     "family": "Инженер-исследователь", "w": 2},
    {"id": 2, "trl": "6–7", "name": "Разработка технологий (ОКР/ОТР)",
     "family": "Инженер-разработчик", "w": 2},
    {"id": 3, "trl": "8–9", "name": "Внедрение в производство",
     "family": "Инженер-технолог", "w": 2},
]

# role_id → TRL placement. Transversal roles (computational/data/management
# dominant) exist at every stage, so they get span (0,9) and no single stage.
TRL_MAP = {
    0: {"stage_id": 0, "span": (0, 4), "family": "Исследователь", "transversal": False},
    1: {"stage_id": 1, "span": (0, 9), "family": "Инженер-исследователь", "transversal": True},
    2: {"stage_id": None, "span": (0, 9), "family": None, "transversal": True},
    3: {"stage_id": 2, "span": (5, 9), "family": "Инженер-разработчик", "transversal": False},
    4: {"stage_id": 0, "span": (0, 4), "family": "Исследователь", "transversal": False},
    5: {"stage_id": 1, "span": (0, 9), "family": "Инженер-исследователь", "transversal": True},
}

# Доказательные результаты (артефакты) per role family — a function of TRL stage.
ARTIFACTS = {
    "Исследователь": ["рецензируемые публикации (статьи/препринты)",
                      "протокол эксперимента и лабораторный журнал",
                      "доклад на конференции", "ВКР/диссертация"],
    "Инженер-исследователь": ["proof-of-concept", "валидированный воспроизводимый метод",
                              "лабораторный прототип (штамм/сборка/пайплайн)",
                              "заявка на патент/РИД"],
    "Инженер-разработчик": ["техническая спецификация и архитектура",
                            "инженерный прототип/биопроцесс/пайплайн",
                            "комплект ОКР/ОТР-документации", "патент",
                            "технологическая карта"],
    "Инженер-технолог": ["производственный регламент", "акт внедрения/валидации",
                         "сертификация (GMP/ISO)", "план масштабирования и дорожная карта"],
}

# Грейд роли (уровень сформированности компетенций) — 5 уровней по модели
# Дрейфуса. Семь карьерных уровней сворачиваются в пять грейдов; разрыв
# компетенций между соседними грейдами значительный.
GRADES = [
    {"id": 0, "name": "Исполнитель", "dreyfus": "Новичок", "levels": [0], "t": 0.0,
     "definition": "выполняет операции по готовому протоколу под контролем"},
    {"id": 1, "name": "Специалист", "dreyfus": "Продвинутый новичок", "levels": [1], "t": 1 / 6,
     "definition": "самостоятельно выполняет типовые задачи, выбирает известный метод"},
    {"id": 2, "name": "Интегратор", "dreyfus": "Компетентный", "levels": [2, 3], "t": 2.5 / 6,
     "definition": "комбинирует методы под новую задачу, ведёт подзадачи, обучает младших"},
    {"id": 3, "name": "Архитектор", "dreyfus": "Опытный", "levels": [4, 5], "t": 4.5 / 6,
     "definition": "проектирует методику/архитектуру, владеет направлением (PI/PE)"},
    {"id": 4, "name": "Создатель", "dreyfus": "Эксперт", "levels": [6], "t": 1.0,
     "definition": "создаёт новые направления и платформы, отвечает за портфель (PP/директор)"},
]

# Требования к компетенциям по стадиям УГТ (спрос) — 7 осей × 4 стадии.
# Значения — целевой уровень компетенции, которого требует стадия.
STAGE_REQUIREMENTS = [
    {"stage": 0, "axes": {"experimental": 3.0, "domain_knowledge": 2.5, "management": 1.0,
                          "professional_texts": 3.0, "data_analysis": 1.5, "computational": 1.5,
                          "t_profile": 1.5}},
    {"stage": 1, "axes": {"experimental": 3.5, "domain_knowledge": 2.5, "management": 1.5,
                          "professional_texts": 2.0, "data_analysis": 3.0, "computational": 2.5,
                          "t_profile": 2.0}},
    {"stage": 2, "axes": {"experimental": 3.0, "domain_knowledge": 2.5, "management": 2.5,
                          "professional_texts": 1.5, "data_analysis": 3.0, "computational": 3.5,
                          "t_profile": 2.0}},
    {"stage": 3, "axes": {"experimental": 2.5, "domain_knowledge": 2.0, "management": 4.0,
                          "professional_texts": 1.0, "data_analysis": 3.0, "computational": 3.0,
                          "t_profile": 2.5}},
]

# Команды по стадиям (роль, грейд, кол-во) — покрывают требования выше.
STAGE_TEAMS = {
    0: [(4, 3, 1), (0, 2, 1), (0, 0, 1), (1, 1, 1), (2, 2, 1)],
    1: [(3, 3, 1), (0, 2, 1), (1, 2, 1), (5, 2, 1), (4, 3, 1), (2, 2, 1)],
    2: [(3, 3, 1), (1, 2, 1), (5, 2, 1), (0, 2, 1), (3, 1, 1), (2, 3, 1)],
    3: [(2, 4, 1), (3, 3, 1), (5, 2, 1), (3, 1, 2), (2, 2, 1), (0, 1, 1)],
}

# Ролевая карта: проектные функции на пересечении (УГТ-стадия × грейд).
# Экспертные грейды = PI / PE / PP.
ROLE_MAP = [
    {"stage": 0, "grade": 0, "title": "лаборант-исследователь, стажёр"},
    {"stage": 0, "grade": 1, "title": "м.н.с., лаборант-исследователь"},
    {"stage": 0, "grade": 2,
     "title": "с.н.с., исследователь-экспериментатор, исследователь-аналитик"},
    {"stage": 0, "grade": 3, "title": "PI — руководитель научной группы, зав. лабораторией"},
    {"stage": 0, "grade": 4, "title": "руководитель научного направления, профессор"},
    {"stage": 1, "grade": 0, "title": "лаборант, техник лаборатории"},
    {"stage": 1, "grade": 1, "title": "инженер-стажёр, м.н.с."},
    {"stage": 1, "grade": 2,
     "title": "инженер-исследователь, биоинформатик-исследователь, инженер по валидации"},
    {"stage": 1, "grade": 3,
     "title": "PI + PE — руководитель R&D-отдела, PI трансляционных исследований"},
    {"stage": 1, "grade": 4, "title": "руководитель R&D-направления"},
    {"stage": 2, "grade": 0, "title": "инженер по испытаниям, техник лаборатории, специалист GxP"},
    {"stage": 2, "grade": 1, "title": "инженер-стажёр, инженер по валидации"},
    {"stage": 2, "grade": 2,
     "title": "инженер-биотехнолог, биоинженер-разработчик, системный биолог"},
    {"stage": 2, "grade": 3, "title": "PE — руководитель ОКР/ОТР, ведущий биоинженер-конструктор"},
    {"stage": 2, "grade": 4, "title": "руководитель инженерного направления, главный конструктор"},
    {"stage": 3, "grade": 0, "title": "сервисный инженер, оператор, инженер-испытатель"},
    {"stage": 3, "grade": 1, "title": "инженер-технолог, инженер по сборке"},
    {"stage": 3, "grade": 2,
     "title": "инженер-технолог, инженер по качеству (QA/QC GMP), инженер-метролог"},
    {"stage": 3, "grade": 3, "title": "PP — главный технолог (GMP), технический директор"},
    {"stage": 3, "grade": 4, "title": "директор по производству, лидер продуктового направления"},
]

# Лестница грейдов сквозных ролей — они работают на всех стадиях УГТ, поэтому
# заданы не как ячейки матрицы, а как вертикальный рост по грейдам.
TRANSVERSAL_LADDER = {
    1: {0: "младший биоинформатик", 1: "биоинформатик", 2: "старший биоинформатик",
        3: "ведущий биоинформатик", 4: "руководитель направления биоинформатики"},
    5: {0: "младший аналитик данных", 1: "аналитик данных", 2: "старший аналитик данных",
        3: "ведущий аналитик данных", 4: "руководитель направления аналитики"},
    2: {0: "ассистент менеджера", 1: "менеджер лаборатории", 2: "старший менеджер лаборатории",
        3: "руководитель лаборатории", 4: "директор лабораторного комплекса"},
}

# ---------------------------------------------------------------------------
# Transition map + growth analysis (derived from ROLES, never hand-written)
# ---------------------------------------------------------------------------

# Leadership is a pseudo-node (NOT a base role): reached when management AND
# domain_knowledge both reach the threshold. "Менеджер лаборатории" (role 2) is
# the base role whose C-level is already closest to this state.
LEADERSHIP_TARGET: dict[str, float] = {"management": 4.0, "domain_knowledge": 4.0}
LEADERSHIP_LABEL = "Руководитель лаборатории"

# An edge A→B exists when the two roles share at least one dominant axis and
# their 7-axis C-level profiles are close (cosine similarity ≥ TRANSITION_MIN_SIM
# and total growth gap ≤ TRANSITION_MAX_GAP). Each role keeps only its
# TRANSITION_MAX_DEGREE closest neighbours, so the map stays sparse instead of
# collapsing into a complete graph on near-identical profiles; a per-role
# fallback below guarantees every role keeps at least one outgoing transition.
TRANSITION_MIN_SIM = 0.5
TRANSITION_MAX_GAP = 20.0
TRANSITION_MAX_DEGREE = 2


def _cosine_sim(a: int, b: int) -> float:
    u = [ROLES[a]["clevel"][x] for x in AXIS_IDS]
    v = [ROLES[b]["clevel"][x] for x in AXIS_IDS]
    dot = sum(x * y for x, y in zip(u, v, strict=True))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def _shared_axes(a: int, b: int) -> list[str]:
    """Dominant axes shared by roles ``a`` and ``b`` (top-first per role ``a``)."""
    return [d for d in ROLES[a]["dominant"] if d in ROLES[b]["dominant"]]


def _shared_axis(a: int, b: int) -> str | None:
    """Top shared dominant axis between roles ``a`` and ``b`` (or ``None``)."""
    shared = _shared_axes(a, b)
    return shared[0] if shared else None


def _gap(src_junior: dict[str, float], tgt: dict[str, float],
         axes: list[str]) -> dict[str, float]:
    return {a: round(max(0.0, tgt[a] - src_junior[a]), 2) for a in axes}


def _gap_score(gap: dict[str, float]) -> float:
    return round(sum(gap.values()), 2)


def _neighbour_rank(source: int, target: int) -> tuple[float, float]:
    gap = _gap(ROLES[source]["junior"], ROLES[target]["clevel"], AXIS_IDS)
    return (-_cosine_sim(source, target), _gap_score(gap))


def _transitions() -> list[dict[str, Any]]:
    n = len(ROLES)
    edges: list[dict[str, Any]] = []
    for s in range(n):
        candidates = [
            t for t in range(n)
            if t != s
            and _shared_axes(s, t)
            and _cosine_sim(s, t) >= TRANSITION_MIN_SIM
            and _gap_score(_gap(ROLES[s]["junior"], ROLES[t]["clevel"], AXIS_IDS))
            <= TRANSITION_MAX_GAP
        ]
        kept = sorted(candidates, key=lambda t: _neighbour_rank(s, t))[
            :TRANSITION_MAX_DEGREE
        ]
        if kept:
            for t in kept:
                edges.append({
                    "source": s, "target": t,
                    "shared_axis": _shared_axes(s, t)[0],
                    "gap": _gap(ROLES[s]["junior"], ROLES[t]["clevel"], AXIS_IDS),
                })
            continue
        best = min(
            (t for t in range(n) if t != s),
            key=lambda t: _neighbour_rank(s, t),
        )
        shared = _shared_axes(s, best)
        edges.append({
            "source": s, "target": best,
            "shared_axis": shared[0] if shared else None,
            "gap": _gap(ROLES[s]["junior"], ROLES[best]["clevel"], AXIS_IDS),
        })
    return edges


def _leadership_edges() -> list[dict[str, Any]]:
    return [
        {"source": s, "target": -1, "shared_axis": None,
         "gap": _gap(ROLES[s]["junior"], LEADERSHIP_TARGET, list(LEADERSHIP_TARGET))}
        for s in range(len(ROLES))
    ]


def _best_switch(role_id: int) -> dict[str, Any]:
    """Best transition for ``role_id``: closest role, leadership as fallback.

    Never returns ``None`` — a role without any role→role transition falls back
    to the leadership pseudo-node (target=-1).
    """
    candidates = [e for e in _transitions() if e["source"] == role_id]
    if not candidates:
        return _leadership_edges()[role_id]
    return min(candidates, key=lambda e: _gap_score(e["gap"]))


def _top_gaps(gap: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(gap.items(), key=lambda kv: kv[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Team composition guide — organised by TRL stage (see STAGE_TEAMS above)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# References + rationale
# ---------------------------------------------------------------------------

REFERENCES = [
    ("SFIA — Skills Framework for the Information Age", "https://sfia-online.org",
     "Международная модель навыков с уровнями ответственности — источник идеи "
     "уровневого роста компетенций."),
    ("ESCO — European Skills, Competences, Occupations", "https://esco.ec.europa.eu",
     "Европейская таксономия профессий и навыков — эталон маппинга навыков на роли."),
    ("O*NET — US Department of Labor", "https://www.onetonline.org",
     "База профессий с требованиями к навыкам и способностям."),
    ("Профессиональные стандарты РФ (Минтруд)", "https://profstandart.rosmintrud.ru",
     "Российские профстандарты для сопоставления ролей с требованиями рынка."),
    ("ФГОС ВО", "https://fgosvo.ru",
     "Образовательные стандарты, определяющие компетенции выпускников."),
    ("ВАК — номенклатура научных специальностей", "https://vak.minobrnauki.gov.ru",
     "Реестр научных специальностей для маппинга исследовательских ролей."),
    ("ГОСТ Р 71726-2024 — Трансфер технологий (УГТ/TRL)", "https://www.rst.gov.ru/portal/gost/home/standarts/technological-readiness",
     "Методика оценки уровня готовности технологий (УГТ 1–9) — источник шкалы "
     "технологической зрелости."),
    ("Кривая обучения (S-кривая)", "https://en.wikipedia.org/wiki/Learning_curve",
     "S-образная сигмоида — общая форма роста навыка: медленный старт, быстрый рост, "
     "плато. Обоснование нелинейного роста компетенций."),
    ("Модель Дрейфуса", "https://en.wikipedia.org/wiki/Dreyfus_model_of_skill_acquisition",
     "Пять стадий освоения навыка (новичок → эксперт) — обоснование стадийного, "
     "неравномерного роста компетенций."),
]

RATIONALE = [
    "Шесть компетенций охватывают полный профиль биологической роли: практическая "
    "экспериментальная работа, предметные знания, управление, научная коммуникация, "
    "количественный анализ и вычисления. Шесть — минимальный набор, при котором "
    "каждая роль описывается уникальной доминантной парой, и роли остаются "
    "различимыми без сотен пересекающихся микро-навыков.",
    "Роль определяется не одним навыком, а двумя ведущими компетенциями — это "
    "позволяет описывать гибридных специалистов (например, биоинформатик = знания + "
    "вычисления), которых одиночная таксономия навыков размывает. Рост от junior к "
    "C-level монотонный, потому что профессионалы накапливают компетенции, а не "
    "теряют их: старший специалист сохраняет практические навыки и добавляет "
    "лидерство и коммуникацию.",
    "Рост компетенций нелинейный: он следует S-образной кривой обучения. Практические "
    "навыки (эксперимент, вычисления) осваиваются рано и выходят на плато, тогда как "
    "управление и научная коммуникация созревают поздно — на senior-уровнях и выше. "
    "Поэтому для каждой компетенции задана своя форма кривой роста, а не общая прямая.",
    "Приведённые значения иллюстративны: они показывают целевой формат отчёта до "
    "того, как полный пайплайн (сбор вакансий, кластеризация, оценка компетенций "
    "через NLP) заполнит реальные числа. Структура при этом не изменится.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Growth shape per competence: exponent p in value = junior + (clevel-junior)·t^p.
# p > 1 → competence grows LATE (management, communication: leadership develops
# at senior levels); p < 1 → grows EARLY (hands-on experimental/computational
# skills are learned fast, then plateau). This models a non-linear, per-axis
# learning curve instead of uniform linear growth.
GROWTH_SHAPE: dict[str, float] = {
    "experimental": 0.7,      # early — hands-on, learned fast then plateaus
    "domain_knowledge": 1.0,  # steady accumulation
    "management": 2.2,        # late — leadership develops at senior levels
    "professional_texts": 1.6,  # late-ish — publishing grows with seniority
    "data_analysis": 0.9,     # early-mid
    "computational": 0.8,     # early — tools learned early
    "t_profile": 1.0,         # steady — breadth accrues evenly with seniority
}


def _grow(junior: float, clevel: float, t: float, axis: str) -> float:
    p = GROWTH_SHAPE.get(axis, 1.0)
    return round(junior + (clevel - junior) * (t ** p), 2)


def _trajectory(role: MockRole) -> list[CareerLevel]:
    n = len(CAREER_LEVELS)
    return [
        {"label": CAREER_LEVELS[i],
         "axes": {a: _grow(role["junior"][a], role["clevel"][a], i / (n - 1), a)
                  for a in AXIS_IDS}}
        for i in range(n)
    ]


def _to_role_model(role: MockRole, axis_labels: dict[str, str]) -> dict[str, Any]:
    return {
        "role_id": role["role_id"],
        "role_label": role["label"],
        "top_job_titles": role["top_titles"],
        "member_count": role["member_count"],
        "characteristics": [
            {
                "characteristic_id": cid,
                "label_ru": axis_labels[cid],
                "proficiency": role["clevel"][cid],
                "top_skills": role["skills"].get(cid, []),
            }
            for cid in AXIS_IDS
        ],
        "competency_levels": [],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --brand:#16324f; --brand2:#24517a; --accent:#2c5f8a; --accent2:#e07a5f;
  --ink:#1e293b; --muted:#64748b; --line:#e2e8f0; --bg:#f5f7fa; --card:#ffffff;
  --good:#15803d; --tint:#eef4fa;
  --shadow:0 1px 2px rgba(16,42,67,.06),0 2px 8px rgba(16,42,67,.05);
  --radius:14px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family:'Inter','Segoe UI',system-ui,-apple-system,'Helvetica Neue',Arial,sans-serif;
       color:var(--ink); margin:0; background:var(--bg); line-height:1.62;
       overflow-x:hidden; -webkit-font-smoothing:antialiased; }
.wrap { max-width:1120px; width:100%; margin:0 auto; padding:40px 32px 80px; }

/* ---------- header ---------- */
header { background:linear-gradient(135deg,var(--brand) 0%,var(--brand2) 60%,#2c5f8a 100%);
         color:#fff; padding:56px 28px 48px; }
header .wrap { padding:0 32px; }
header h1 { margin:0 0 10px; font-size:28px; font-weight:700;
            letter-spacing:-.01em; line-height:1.25; }
header .sub { margin:0; opacity:.85; font-size:15px; }
.badges { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.badge { display:inline-block; padding:4px 12px; border-radius:999px; font-size:12px;
         font-weight:600; background:rgba(255,255,255,.14);
         border:1px solid rgba(255,255,255,.25); }

/* ---------- sections ---------- */
section { margin:44px 0; scroll-margin-top:24px; }
h2 { font-size:21px; font-weight:700; letter-spacing:-.01em; margin:0 0 20px;
     padding:0 0 10px; border-bottom:1px solid var(--line); position:relative; }
h2::after { content:''; position:absolute; left:0; bottom:-1px; width:56px; height:3px;
            background:var(--accent); border-radius:2px; }
h3 { font-size:16px; margin:0 0 12px; font-weight:700; }
h4 { font-size:13px; margin:0 0 8px; color:var(--accent); font-weight:700; }
p { margin:0 0 14px; }
.lead { font-size:15px; color:var(--muted); max-width:760px; }

/* ---------- KPI cards ---------- */
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:16px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
       padding:22px 18px; text-align:center; box-shadow:var(--shadow);
       border-top:3px solid var(--accent); }
.kpi .v { font-size:30px; font-weight:800; color:var(--brand2);
          letter-spacing:-.02em; line-height:1.1; }
.kpi .l { font-size:12px; color:var(--muted); margin-top:6px; }

/* ---------- cards & tables ---------- */
.card { background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
        padding:24px; box-shadow:var(--shadow); }
table { width:100%; border-collapse:collapse; background:var(--card);
        border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:14px; }
th { background:var(--tint); color:var(--brand2); font-weight:700; text-align:left;
     padding:12px 16px; border-bottom:1px solid var(--line); }
td { padding:11px 16px; border-bottom:1px solid var(--line); vertical-align:top; }
tbody tr:nth-child(even) { background:#fbfcfe; }
tbody tr:last-child td { border-bottom:none; }
tbody tr:hover { background:#f1f6fb; }
.ok { color:var(--good); font-weight:700; }

/* ---------- roles ---------- */
.role { margin-bottom:44px; }
.role-head { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:16px; }
.role-head h3 { margin:0; font-size:19px; }
.role-head .titles { font-size:13px; color:var(--muted); }
.role-head .dominant { display:flex; gap:6px; flex-wrap:wrap; }
.badge-dom { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11.5px;
             font-weight:600; background:var(--tint); color:var(--brand2);
             border:1px solid #d5e3f0; }

/* skills as chips */
.skill-chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.skill-group { flex:1 1 100%; margin-bottom:10px; }
.skill-group h4 { margin:0 0 6px; }
.chip { display:inline-block; padding:4px 11px; border-radius:999px; font-size:12px;
        background:var(--tint); color:var(--ink); border:1px solid #dce7f2; margin:0 4px 4px 0; }

/* skill tree */
.skill-tree { }
.st-branch { display:flex; align-items:flex-start; gap:12px; padding:7px 0 7px 14px;
             border-left:2px solid var(--line); margin-left:8px; }
.st-branch.st-dom { border-left-color:var(--accent2); }
.st-label { font-weight:600; font-size:13px; width:220px; flex-shrink:0; color:var(--ink); }
.st-branch.st-dom .st-label { color:var(--brand2); }
.st-prof { color:var(--accent); font-size:11px; font-weight:700; }
.st-chips { display:flex; flex-wrap:wrap; gap:2px; }

/* charts */
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
.grid2 .cell { min-width:0; }
img { max-width:100%; height:auto; display:block; }
.frame { border:1px solid var(--line); border-radius:12px; padding:10px; background:#fcfdff; }
.full img { width:100%; }

.growth { margin:0; padding-left:18px; font-size:13px; }
.growth li { margin:4px 0; }
.note { font-size:12.5px; color:var(--muted); margin-top:10px; line-height:1.5; }
.refs li { margin:8px 0; font-size:14px; }
.refs a { color:var(--accent); font-weight:600; text-decoration:none; }
.refs a:hover { text-decoration:underline; }

/* role constructor */
.csliders { display:flex; flex-direction:column; gap:14px; margin:12px 0; }
.csl { display:flex; align-items:center; gap:12px; }
.csl label { width:210px; font-size:13px; flex-shrink:0; }
.csl input[type=range] { flex:1; accent-color:var(--accent); }
.cval { width:34px; text-align:right; font-weight:700; color:var(--brand2);
        font-variant-numeric:tabular-nums; }
.constructor-out { font-size:15px; font-weight:600; margin-top:18px; padding:14px 16px;
                   background:var(--tint); border-radius:10px;
                   border:1px solid #d5e3f0; line-height:1.5; }
.constructor-out b { color:var(--brand2); }

/* team constructor */
.troles { display:flex; flex-direction:column; gap:10px; }
.trole { display:flex; align-items:center; gap:10px; padding:9px 12px;
         background:#fbfcfe; border:1px solid var(--line); border-radius:10px; }
.tname { font-weight:600; font-size:13px; flex:1; }
.tdom { font-size:11.5px; color:var(--muted); width:150px; flex-shrink:0; }
.tbtn { width:26px; height:26px; border-radius:7px; border:1px solid var(--line);
        background:var(--card); font-weight:700; cursor:pointer; color:var(--brand2);
        font-size:15px; line-height:1; }
.tbtn:hover { background:var(--tint); border-color:var(--accent); }
.tcnt { width:22px; text-align:center; font-weight:700; font-variant-numeric:tabular-nums; }
.presets { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
.preset { padding:6px 13px; border-radius:999px; border:1px solid var(--line);
          background:var(--card); cursor:pointer; font-size:12.5px; font-weight:600;
          color:var(--accent); }
.preset:hover { background:var(--accent); color:#fff; border-color:var(--accent); }
.cov { display:flex; align-items:center; gap:8px; font-size:13.5px; margin:5px 0; }
.cov .cok { color:var(--good); font-weight:700; }
.cov .cno { color:var(--muted); }
.warn { margin-top:12px; padding:10px 14px; background:#fdf3e7; border:1px solid #f5d9b8;
        border-radius:10px; font-size:13px; color:#9a5b00; }

/* team composer (drag & drop) */
.spec-row { display:flex; align-items:center; gap:6px; margin-bottom:8px; flex-wrap:wrap; }
.spec-role { font-size:12.5px; font-weight:600; width:170px; flex-shrink:0; }
.spec { display:inline-block; padding:4px 10px; border-radius:999px; font-size:11.5px;
        background:var(--tint); border:1px dashed #b8c9d8; cursor:grab; user-select:none; }
.spec:hover { background:#dce9f4; border-color:var(--accent); }
.spec:active { cursor:grabbing; }
.team-zone { min-height:110px; border:2px dashed #c3d2df; border-radius:12px; padding:12px;
             background:#fbfcfe; margin-bottom:12px; }
.team-zone .trl-chip { margin:0 6px 6px 0; }
.team-zone .rm { color:#c0392b; text-decoration:none; font-weight:700; margin-left:2px; }

/* TRL map */
.trl-bar { display:flex; gap:4px; margin:8px 0 4px; }
.trl-seg { border:1px solid var(--line); border-radius:10px; padding:10px 12px;
           background:#fbfcfe; }
.trl-band { font-size:12px; font-weight:800; color:var(--brand2); }
.trl-name { font-size:12px; color:var(--ink); font-weight:600; }
.trl-family { font-size:12px; color:var(--accent); font-weight:700; margin-bottom:6px; }
.trl-chips { display:flex; flex-wrap:wrap; gap:4px; }
.trl-chip { font-size:11px; background:var(--tint); border:1px solid #d5e3f0;
            border-radius:999px; padding:2px 8px; }
.trl-span { color:var(--muted); }
.trl-transv { margin-top:10px; font-size:13px; color:#9a5b00; padding:10px 14px;
              background:#fdf3e7; border:1px solid #f5d9b8; border-radius:10px; }
.trl-arts { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.trl-art { background:#fbfcfe; border:1px solid var(--line); border-radius:10px;
           padding:12px 14px; }
@media (max-width: 860px) { .trl-arts { grid-template-columns:1fr; } }

/* role map matrix */
.table-scroll { overflow-x:auto; }
table.rolemap th, table.rolemap td { font-size:12.5px; vertical-align:top; }
table.rolemap th { white-space:normal; }
.th-sub { font-weight:500; color:var(--muted); font-size:11px; }
.th-fam { font-weight:700; color:var(--accent); font-size:11.5px; }

/* stage cards */
.stage-card { background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
              padding:20px; box-shadow:var(--shadow); margin-bottom:20px; }
.stage-card h4 { font-size:14px; margin:0 0 10px; }

/* footer */
footer { text-align:center; padding:32px 20px 48px; color:var(--muted); font-size:13px;
         border-top:1px solid var(--line); }

/* vacancy example */
.vacancy { border:1px solid #d5e3f0; background:#f8fbfe; border-radius:12px;
           padding:14px 18px; margin:0 0 18px; }
.vacancy h4 { margin:0 0 6px; }
.vacancy-title { font-weight:700; font-size:15px; color:var(--brand2); margin-bottom:4px; }
.vacancy-desc { font-size:13.5px; margin:0 0 10px; white-space:pre-line; }
.vacancy-src { font-size:12px; color:var(--muted); margin-top:6px; }

@media (max-width: 860px) { .grid2 { grid-template-columns:1fr; } }
"""


def _kpi_html() -> str:
    return "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{lab}</div></div>'
        for v, lab in KPIS
    )


def _metrics_html() -> str:
    rows = "".join(
        f'<tr><td>{name}</td><td>{value}</td><td>{target}</td>'
        f'<td class="ok">✓</td></tr>'
        for name, value, target in METRICS
    )
    return (
        "<table><tr><th>Метрика</th><th>Значение</th><th>Цель</th><th></th></tr>"
        f"{rows}</table>"
    )


def _metric_explain_html() -> str:
    rows = "".join(
        f"<tr><td><b>{name}</b></td><td>{how}</td><td>{proves}</td></tr>"
        for name, how, proves in METRIC_EXPLANATIONS
    )
    return (
        '<h3>Как считаются метрики и что они доказывают</h3>'
        "<table><tr><th>Метрика</th><th>Как считаем</th><th>Что доказывает</th></tr>"
        f"{rows}</table>"
    )


def _animated_spider_html(role: MockRole, axis_labels: dict[str, str],
                          include_js: bool) -> str:
    """Interactive animated spider (junior → C-level morph) via Plotly.

    Axes are placed clockwise from the top (rotation=90, direction="clockwise")
    to match the static matplotlib spider charts, so the same competence sits at
    the same position across every visualisation.
    """
    import plotly.graph_objects as go

    levels = _trajectory(role)
    n_axes = len(AXIS_IDS)
    theta_deg = [i * (360 / n_axes) for i in range(n_axes)]
    theta_labels = [axis_labels[a] for a in AXIS_IDS]
    theta = theta_deg + [360]

    def _r(lvl: dict) -> list[float]:
        return [lvl["axes"][a] for a in AXIS_IDS] + [lvl["axes"][AXIS_IDS[0]]]

    frames = [
        go.Frame(
            data=[go.Scatterpolar(
                r=_r(lvl), theta=theta, fill="toself", name=lvl["label"],
                line={"color": "#2C5F8A"}, opacity=0.85,
            )],
            name=lvl["label"],
        )
        for lvl in levels
    ]

    slider_steps = [
        {"args": [[lvl["label"]],
                  {"frame": {"duration": 500, "redraw": True},
                   "mode": "immediate",
                   "transition": {"duration": 400, "easing": "cubic-in-out"}}],
         "label": lvl["label"], "method": "animate"}
        for lvl in levels
    ]

    fig = go.Figure(
        data=[go.Scatterpolar(r=_r(levels[0]), theta=theta, fill="toself",
                              name="", line={"color": "#2C5F8A"}, opacity=0.85)],
        frames=frames,
        layout=go.Layout(
            title={"text": f"Компетентностный рост: {role['label']}",
                       "font": {"size": 14}},
            polar={
                "radialaxis": {"visible": True, "range": [1, 5],
                                "tickvals": [1, 2, 3, 4, 5]},
                "angularaxis": {
                    "direction": "clockwise", "rotation": 90,
                    "tickmode": "array", "tickvals": theta_deg, "ticktext": theta_labels,
                    "tickfont": {"size": 11},
                },
            },
            sliders=[{
                "active": 0,
                "steps": slider_steps,
                "currentvalue": {"prefix": "Уровень: ", "font": {"size": 12}},
                "pad": {"t": 40},
            }],
            height=440,
            margin={"l": 70, "r": 70, "t": 70, "b": 40},
        ),
    )
    return fig.to_html(include_plotlyjs="cdn" if include_js else False,
                       full_html=False, div_id=f"spider_anim_{role['role_id']}")


def _role_constructor_html(axis_labels: dict[str, str]) -> str:
    """Interactive role constructor: one slider per hard axis drives a spider;
    JS matches the drawn profile to the nearest role (shape) and career level
    (magnitude)."""
    import json

    import plotly.graph_objects as go

    axis_order = [axis_labels[a] for a in AXIS_IDS]
    roles_data = [
        {
            "label": r["label"],
            "clevel": [r["clevel"][a] for a in AXIS_IDS],
            "trajectory": [[lvl["axes"][a] for a in AXIS_IDS] for lvl in _trajectory(r)],
        }
        for r in ROLES
    ]

    theta_deg = [i * (360 / len(AXIS_IDS)) for i in range(len(AXIS_IDS))]
    fig = go.Figure(
        data=[go.Scatterpolar(
            r=[3.0] * len(AXIS_IDS) + [3.0], theta=theta_deg + [360], fill="toself",
            line={"color": "#2C5F8A"}, opacity=0.85,
        )],
        layout=go.Layout(
            polar={
                "radialaxis": {"visible": True, "range": [1, 5], "tickvals": [1, 2, 3, 4, 5]},
                "angularaxis": {"direction": "clockwise", "rotation": 90,
                                 "tickmode": "array", "tickvals": theta_deg,
                                 "ticktext": axis_order, "tickfont": {"size": 11}},
            },
            showlegend=False, height=420,
            margin={"l": 70, "r": 70, "t": 30, "b": 40},
        ),
    )
    chart = fig.to_html(include_plotlyjs=False, full_html=False,
                        div_id="constructor_radar")

    sliders = "".join(
        f'<div class="csl"><label>{axis_labels[a]}</label>'
        f'<input type="range" min="1" max="5" step="0.1" value="3" data-i="{i}">'
        f'<span class="cval" data-out="{i}">3.0</span></div>'
        for i, a in enumerate(AXIS_IDS)
    )

    js_data = json.dumps(
        {"roles": roles_data, "levels": CAREER_LEVELS}, ensure_ascii=False
    )

    js = """
<script>
(function(){
  var DATA = __DATA__;
  var n = __N__;
  var vals = Array(n).fill(3);

  function match(v) {
    var mean = v.reduce(function(a,b){return a+b;},0)/n;
    var vc = v.map(function(x){return x-mean;});
    var vnorm = Math.sqrt(vc.reduce(function(a,b){return a+b*b;},0));
    var corrs = DATA.roles.map(function(role){
      var p = role.clevel;
      var pmean = p.reduce(function(a,b){return a+b;},0)/n;
      var pc = p.map(function(x){return x-pmean;});
      var pnorm = Math.sqrt(pc.reduce(function(a,b){return a+b*b;},0));
      if (vnorm < 0.3 || pnorm < 1e-9) return 0;
      return vc.reduce(function(s,x,j){return s+x*pc[j];},0)/(vnorm*pnorm);
    });
    var best = 0;
    corrs.forEach(function(c,i){ if (c > corrs[best]) best = i; });
    var t = 0.35;
    var exps = corrs.map(function(c){ return Math.exp(c/t); });
    var esum = exps.reduce(function(a,b){return a+b;},0);
    var conf = Math.round(exps[best]/esum*100);

    var traj = DATA.roles[best].trajectory;
    var bl = 0, bd = 1e9;
    traj.forEach(function(lv,i){
      var d = Math.sqrt(v.reduce(function(s,x,j){return s+Math.pow(x-lv[j],2);},0));
      if (d < bd) { bd = d; bl = i; }
    });

    return {best:best, conf:conf, level:bl, flat:vnorm < 0.3};
  }

  function update() {
    var r = vals.concat(vals[0]);
    var th = [];
    for (var i=0;i<=n;i++) th.push(i*(360/n));
    Plotly.restyle('constructor_radar', {r:[r], theta:[th]}, [0]);
    var m = match(vals);
    var out = document.getElementById('constructor_out');
    if (m.flat) {
      out.innerHTML = 'Универсальный профиль — нет выраженной доминантной пары компетенций';
    } else {
      out.innerHTML = 'Ближайшая роль: <b>' + DATA.roles[m.best].label +
        '</b> (сходство ' + m.conf + '%) · Уровень: <b>' + DATA.levels[m.level] + '</b>';
    }
  }

  document.querySelectorAll('#constructor_wrap input[type=range]').forEach(function(el){
    el.addEventListener('input', function(e){
      var i = +e.target.dataset.i;
      vals[i] = +e.target.value;
      document.querySelector('[data-out="'+i+'"]').textContent = (+e.target.value).toFixed(1);
      update();
    });
  });
  update();
})();
</script>
""".replace("__DATA__", js_data).replace("__N__", str(len(AXIS_IDS)))

    return f"""
<div class="card">
  <div class="grid2">
    <div class="cell">
      <h3 style="margin-top:0">Профиль компетенций</h3>
      {chart}
    </div>
    <div class="cell">
      <h3 style="margin-top:0">Задайте уровень каждой компетенции</h3>
      <div id="constructor_wrap" class="csliders">{sliders}</div>
      <div id="constructor_out" class="constructor-out"></div>
      <p class="note">Двигайте ползунки — профиль на графике, ближайшая роль и
      карьерный уровень пересчитываются мгновенно. Форма профиля определяет роль,
      общая величина — уровень.</p>
    </div>
  </div>
</div>
{js}
"""


def _team_constructor_html(axis_labels: dict[str, str]) -> str:
    """Drag-and-drop team composer per УГТ stage, with size-tier presets.

    The team is a multi-level composition: Минимальный (1–2 лидера) →
    Стандартный → Полный. Coverage is computed as the max value per axis across
    team members, compared against the stage requirement."""
    import json

    role_names = [r["label"] for r in ROLES]
    grade_names = [g["name"] for g in GRADES]
    axis_names = [axis_labels[a] for a in AXIS_IDS]

    contrib: dict[str, list[float]] = {}
    for rid in range(len(ROLES)):
        for gid in range(len(GRADES)):
            t = GRADES[gid]["t"]
            contrib[f"{rid},{gid}"] = [
                round(_grow(ROLES[rid]["junior"][a], ROLES[rid]["clevel"][a], t, a), 2)
                for a in AXIS_IDS
            ]

    stages = []
    for s in TRL_STAGES:
        sid = s["id"]
        req = [STAGE_REQUIREMENTS[sid]["axes"][a] for a in AXIS_IDS]
        presets = {}
        for tier, min_grade in (("min", 3), ("std", 2), ("full", -1)):
            pairs = [
                [rid, gid] for rid, gid, cnt in STAGE_TEAMS[sid]
                for _ in range(cnt) if GRADES[gid]["id"] >= min_grade
            ]
            presets[tier] = pairs
        stages.append({"label": f"УГТ {s['trl']} · {s['family']}", "req": req,
                       "presets": presets})

    js_data = json.dumps({
        "roles": role_names, "grades": grade_names, "axes": axis_names,
        "contrib": contrib, "stages": stages,
    }, ensure_ascii=False)

    palette = ""
    for rid in range(len(ROLES)):
        chips = "".join(
            f'<span class="spec" draggable="true" data-role="{rid}" data-grade="{gid}">'
            f'{grade_names[gid]}</span>'
            for gid in range(len(GRADES))
        )
        palette += (
            f'<div class="spec-row"><span class="spec-role">{role_names[rid]}</span>'
            f'{chips}</div>'
        )

    stage_btns = "".join(
        f'<button class="preset" data-stage="{i}">{s["label"]}</button>'
        for i, s in enumerate(stages)
    )

    js = """
<script>
(function(){
  var DATA = __DATA__;
  var team = [];
  var stage = 0;

  function coverage() {
    var sup = DATA.axes.map(function(){ return 0; });
    team.forEach(function(m){
      var key = m[0] + ',' + m[1];
      DATA.contrib[key].forEach(function(v, i){ if (v > sup[i]) sup[i] = v; });
    });
    return sup;
  }

  function render() {
    var zone = document.getElementById('composer_team');
    if (team.length === 0) {
      zone.innerHTML = '<span style="color:var(--muted)">Перетащите специалистов сюда</span>';
    } else {
      zone.innerHTML = team.map(function(m, idx){
        return '<span class="trl-chip">' + DATA.roles[m[0]] + ' · ' + DATA.grades[m[1]] +
          ' <a class="rm" href="#" data-rm="' + idx + '">×</a></span>';
      }).join('');
    }
    var sup = coverage();
    var req = DATA.stages[stage].req;
    var html = '<b>Покрытие для «' + DATA.stages[stage].label + '»</b><br>';
    var covered = 0;
    DATA.axes.forEach(function(name, i){
      var ok = sup[i] >= req[i] - 0.01;
      if (ok) covered++;
      html += '<div class="cov"><span class="' + (ok ? 'cok' : 'cno') + '">' +
        (ok ? '✓' : '—') + '</span> ' + name +
        ' <span style="color:#8aa0b8;font-size:12px">(' + sup[i].toFixed(1) + ' / ' +
        req[i].toFixed(1) + ')</span></div>';
    });
    html += '<div style="margin-top:6px"><b>' + team.length + '</b> чел. · покрыто ' +
      covered + ' из ' + DATA.axes.length + '</div>';
    document.getElementById('composer_cov').innerHTML = html;
  }

  function add(role, grade) {
    team.push([role, grade]);
    render();
  }

  document.querySelectorAll('.spec').forEach(function(el){
    el.addEventListener('dragstart', function(e){
      e.dataTransfer.setData('text/plain', el.dataset.role + ',' + el.dataset.grade);
    });
    el.addEventListener('click', function(){
      add(+el.dataset.role, +el.dataset.grade);
    });
  });

  var zone = document.getElementById('composer_team');
  zone.addEventListener('dragover', function(e){ e.preventDefault(); });
  zone.addEventListener('drop', function(e){
    e.preventDefault();
    var parts = e.dataTransfer.getData('text/plain').split(',');
    if (parts.length === 2) add(+parts[0], +parts[1]);
  });
  zone.addEventListener('click', function(e){
    if (e.target.getAttribute('data-rm') !== null) {
      e.preventDefault();
      team.splice(+e.target.getAttribute('data-rm'), 1);
      render();
    }
  });

  document.querySelectorAll('[data-stage]').forEach(function(btn){
    btn.addEventListener('click', function(){ stage = +btn.dataset.stage; render(); });
  });
  document.querySelectorAll('[data-tier]').forEach(function(btn){
    btn.addEventListener('click', function(){
      team = DATA.stages[stage].presets[btn.dataset.tier].map(function(p){ return p; });
      render();
    });
  });

  render();
})();
</script>
""".replace("__DATA__", js_data)

    return f"""
<div class="card">
  <h3 style="margin-top:0">Соберите команду под стадию УГТ</h3>
  <div class="presets">Стадия: {stage_btns}</div>
  <div class="presets" style="margin-top:8px">
    <span class="tdom" style="width:auto">Размер команды:</span>
    <button class="preset" data-tier="min">Минимальный (1–2)</button>
    <button class="preset" data-tier="std">Стандартный</button>
    <button class="preset" data-tier="full">Полный</button>
  </div>
  <div class="grid2" style="margin-top:14px">
    <div class="cell">
      <h4>Специалисты — перетащите в команду (или кликните)</h4>
      <div class="palette">{palette}</div>
    </div>
    <div class="cell">
      <h4>Ваша команда</h4>
      <div class="team-zone" id="composer_team"></div>
      <div id="composer_cov" class="constructor-out"></div>
    </div>
  </div>
</div>
{js}
"""


def _roles_table_html(models: list[dict[str, Any]], skills_by_role: dict[int, str]) -> str:
    rows = ""
    for m in models:
        titles = ", ".join(m["top_job_titles"][:3])
        trl = _trl_badge(m["role_id"])
        rows += (
            f"<tr><td><b>{m['role_label']}</b></td>"
            f"<td>{trl}</td>"
            f"<td>{m['member_count']}</td>"
            f"<td>{skills_by_role[m['role_id']]}</td>"
            f"<td>{titles}</td></tr>"
        )
    return (
        "<table><tr><th>Роль</th><th>УГТ/TRL</th><th>Вакансий</th><th>Навыков</th>"
        f"<th>Топ названий должностей</th></tr>{rows}</table>"
    )


def _skill_tree_html(role: MockRole, axis_labels: dict[str, str]) -> str:
    """Skill tree as a clean indented HTML list: competences as branches, skills
    as chips. Dominant competences are highlighted — the tree grows along the
    role's signature pair."""
    dom = set(role["dominant"])
    branches = ""
    for a in AXIS_IDS:
        chips = "".join(
            f'<span class="chip">{s}</span>' for s in role["skills"].get(a, [])
        )
        cls = " st-dom" if a in dom else ""
        prof = role["clevel"][a]
        branches += (
            f'<div class="st-branch{cls}">'
            f'<span class="st-label">{axis_labels[a]} '
            f'<span class="st-prof">{prof:.1f}</span></span>'
            f'<div class="st-chips">{chips}</div>'
            f'</div>'
        )
    return f'<div class="skill-tree">{branches}</div>'


def _transition_network_html(axis_labels: dict[str, str], include_js: bool) -> str:
    """Interactive transition network: nodes = roles + leadership, edges = swaps.

    Click a role node to see (below the chart) which roles it can swap into and
    which competences must grow for each swap.
    """
    import json
    import math

    import plotly.graph_objects as go

    n = len(ROLES)
    pos = {}
    for i in range(n):
        a = math.radians(90 + i * (360 / n))
        pos[i] = (math.cos(a), math.sin(a))
    pos[-1] = (0.0, 0.0)

    # edges
    edge_x: list[float] = []
    edge_y: list[float] = []
    for e in _transitions():
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    lead_x: list[float] = []
    lead_y: list[float] = []
    for e in _leadership_edges():
        x0, y0 = pos[e["source"]]
        lead_x += [x0, 0.0, None]
        lead_y += [y0, 0.0, None]

    node_x = [pos[i][0] for i in range(n)]
    node_y = [pos[i][1] for i in range(n)]
    node_names = [r["label"] for r in ROLES]
    edge_targets = {s: [e["target"] for e in _transitions() if e["source"] == s]
                    for s in range(n)}
    hover = []
    for r in ROLES:
        dom = " + ".join(axis_labels[d] for d in r["dominant"])
        conn = ", ".join(ROLES[t]["label"] for t in edge_targets[r["role_id"]])
        hover.append(f"{r['label']}<br>Доминанты: {dom}<br>Связан с: {conn}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line={"color": "#9AA5B1", "width": 1.5}, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=lead_x, y=lead_y, mode="lines",
        line={"color": "#E07A5F", "width": 1.5, "dash": "dot"}, hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker={"size": 30, "color": "#2C5F8A", "line": {"color": "white", "width": 2}},
        text=hover, hoverinfo="text", showlegend=False,
        customdata=list(range(n)),
    ))
    fig.add_trace(go.Scatter(
        x=[0.0], y=[0.0], mode="markers",
        marker={"size": 46, "color": "#E07A5F", "line": {"color": "white", "width": 2},
                   "symbol": "star"},
        text=LEADERSHIP_LABEL, hoverinfo="text", showlegend=False,
    ))

    # node labels
    for i in range(n):
        a = math.radians(90 + i * (360 / n))
        lx, ly = 1.42 * math.cos(a), 1.42 * math.sin(a)
        ha = "center" if abs(math.cos(a)) < 0.32 else ("left" if math.cos(a) > 0 else "right")
        va = "middle" if abs(math.sin(a)) < 0.32 else ("bottom" if math.sin(a) > 0 else "top")
        fig.add_annotation(x=lx, y=ly, text=node_names[i], showarrow=False,
                           font={"size": 12, "color": "#1e293b"},
                           xanchor=ha, yanchor=va)
    fig.add_annotation(x=0, y=-0.24, text=LEADERSHIP_LABEL, showarrow=False,
                       font={"size": 12, "color": "#8A5A00"})

    fig.update_layout(
        height=520, showlegend=False,
        margin={"t": 30, "l": 40, "r": 40, "b": 30},
        xaxis={"visible": False, "range": [-2.1, 2.1]},
        yaxis={"visible": False, "range": [-1.85, 1.85]},
        title={"text": "Карта перехода между ролями (нажмите на роль)", "font": {"size": 14}},
        clickmode="event+select",
    )

    # swap data for the click handler
    swaps = {}
    for r in ROLES:
        entries = []
        for e in _transitions():
            if e["source"] == r["role_id"]:
                target = ROLES[e["target"]]["label"]
                tops = _top_gaps(e["gap"], 3)
                grow = ", ".join(f"{axis_labels[a]} +{d:.1f}" for a, d in tops)
                entries.append({"to": target,
                                "shared": axis_labels.get(e["shared_axis"], "—"),
                                "grow": grow})
        lead = _leadership_edges()[r["role_id"]]
        tops = _top_gaps(lead["gap"], 2)
        entries.append({"to": LEADERSHIP_LABEL, "shared": "—",
                        "grow": ", ".join(f"{axis_labels[a]} +{d:.1f}" for a, d in tops)})
        swaps[r["label"]] = entries
    js_data = json.dumps({"names": node_names, "swaps": swaps}, ensure_ascii=False)

    js = """
<script>
(function(){
  var DATA = __DATA__;
  var gd = document.getElementById('transition_net');
  gd.on('plotly_click', function(ev){
    var pt = ev.points[0];
    if (pt.customdata === undefined || pt.customdata === null) return;
    var name = DATA.names[pt.customdata];
    var swaps = DATA.swaps[name] || [];
    var html = '<b>' + name + '</b> — куда можно перейти:<br>';
    swaps.forEach(function(s){
      html += '<div style="margin:6px 0 0 6px">→ <b>' + s.to + '</b><br>' +
        '<span style="font-size:12px;color:#64748b">через «' + s.shared +
        '» · растить: ' + s.grow + '</span></div>';
    });
    document.getElementById('transition_out').innerHTML = html;
  });
})();
</script>
""".replace("__DATA__", js_data)

    chart = fig.to_html(include_plotlyjs="cdn" if include_js else False,
                        full_html=False, div_id="transition_net")
    return f"""
<div class="card">
  {chart}
  <div id="transition_out" class="constructor-out" style="margin-top:12px">
    Нажмите на роль, чтобы увидеть возможные переходы и какие компетенции растить.
  </div>
</div>
{js}
"""


def _skill_rows(role: MockRole, axis_labels: dict[str, str]) -> str:
    groups = ""
    for cid in AXIS_IDS:
        chips = "".join(
            f'<span class="chip">{s}</span>' for s in role["skills"].get(cid, [])
        )
        groups += (
            f'<div class="skill-group"><h4>{axis_labels[cid]}</h4>'
            f'<div>{chips}</div></div>'
        )
    return f'<div class="skill-chips">{groups}</div>'


def _vacancy_example_html(role: MockRole, axis_labels: dict[str, str]) -> str:
    """Example vacancy for the role.

    Prefers a real vacancy from the pipeline (``role["example_vacancy"]``,
    attached by ``scripts.generate_report``); otherwise synthesizes a short
    sample from the role's own fields.
    """
    vacancy = role.get("example_vacancy") or {}
    if vacancy.get("title") or vacancy.get("description"):
        title = str(vacancy.get("title") or role["label"])
        description = str(vacancy.get("description") or "")
        skills = list(vacancy.get("skills") or [])
        chips = "".join(
            f'<span class="chip">{html.escape(s)}</span>' for s in skills
        )
        title_html = html.escape(title[:1].upper() + title[1:])
        return f"""
<div class="vacancy">
  <h4>Пример вакансии</h4>
  <div class="vacancy-title">Вакансия: {title_html}</div>
  <div class="vacancy-desc">{html.escape(description)}</div>
  <div>{chips}</div>
  <div class="vacancy-src">Реальная вакансия из собранного корпуса</div>
</div>
"""
    title = (role["top_titles"] or [role["label"]])[0]
    title = title[:1].upper() + title[1:]
    dom_labels = [
        axis_labels[d] for d in role["dominant"] if d in role["skills"]
    ]
    skills: list[str] = []
    for d in role["dominant"]:
        for s in role["skills"].get(d, []):
            if s not in skills:
                skills.append(s)
    skills = skills[:8]
    median = (role.get("experience") or {}).get("median")

    sentences = [f"Ищем <b>{title}</b>."]
    if median is not None:
        sentences.append(
            f"Требуемый опыт — около {median:.1f} лет, ключевые компетенции — "
            f"{' и '.join(dom_labels)}."
        )
    else:
        sentences.append(f"Ключевые компетенции — {' и '.join(dom_labels)}.")
    sentences.append(
        f"Образец сформирован по {role['member_count']} похожим вакансиям рынка."
    )
    chips = "".join(f'<span class="chip">{s}</span>' for s in skills)
    return f"""
<div class="vacancy">
  <h4>Пример вакансии</h4>
  <div class="vacancy-title">Вакансия: {title}</div>
  <p class="vacancy-desc">{' '.join(sentences)}</p>
  <div>{chips}</div>
</div>
"""


def _role_html(role: MockRole, axis_labels: dict[str, str],
               charts: dict[str, str | None]) -> str:
    rid = role["role_id"]
    spider = charts.get(f"spider_{rid}")
    soft = charts.get(f"soft_{rid}")
    experience = charts.get(f"experience_{rid}")
    skills = charts.get(f"skills_{rid}")
    animated = _animated_spider_html(role, axis_labels, include_js=False)
    vacancy = _vacancy_example_html(role, axis_labels)
    skilltree = _skill_tree_html(role, axis_labels)
    titles = ", ".join(role["top_titles"])
    dominant_badges = "".join(
        f'<span class="badge-dom">{axis_labels[d]}</span>' for d in role["dominant"]
    )
    trl_badge = f'<span class="badge-dom">{_trl_badge(role["role_id"])}</span>'

    def _diagram(chart: str | None) -> str:
        if chart is None:
            return '<p class="note">нет данных</p>'
        return f'<div class="frame">{chart}</div>'

    return f"""
<div class="role card">
  <div class="role-head">
    <h3>{role['label']}</h3>
    <span class="dominant">{dominant_badges}{trl_badge}</span>
    <span class="titles">{role['member_count']} вакансий · {titles}</span>
  </div>
  {vacancy}
  <div class="grid2">
    <div class="cell">
      <h3 style="margin-top:0">Профиль компетенций (C-level)</h3>
      {_diagram(spider)}
    </div>
    <div class="cell">
      <h3 style="margin-top:0">Компетентностный рост роли (Junior → C-level)</h3>
      {animated}
      <p class="note">Передвигайте ползунок — видно, как меняется форма профиля
      по мере роста.</p>
    </div>
  </div>
  <h3 style="margin-top:22px">Мягкие компетенции, опыт и навыки</h3>
  <div class="grid2">
    <div class="cell"><h4>Мягкие компетенции</h4>{_diagram(soft)}</div>
    <div class="cell"><h4>Требуемый опыт</h4>{_diagram(experience)}</div>
    <div class="cell"><h4>Ключевые навыки</h4>{_diagram(skills)}</div>
  </div>
  <h3>Дерево навыков</h3>
  {skilltree}
  <p class="note">Доминантные компетенции выделены цветом — это «сигнатурные» навыки
  роли. Число рядом с компетенцией — уровень владения (1–5).</p>
</div>
"""


def _growth_html(role: MockRole, axis_labels: dict[str, str]) -> str:
    rid = role["role_id"]
    excel = _top_gaps(_gap(role["junior"], role["clevel"], AXIS_IDS))
    best = _best_switch(rid)
    lead = [(a, LEADERSHIP_TARGET[a] - role["junior"][a])
            for a in ("management", "domain_knowledge")]
    lead = sorted(lead, key=lambda kv: kv[1], reverse=True)

    def _li(pairs: list[tuple[str, float]]) -> str:
        return "".join(
            f"<li><b>{axis_labels[a]}</b> — +{d:.1f}</li>" for a, d in pairs
        )

    best = _best_switch(rid)
    switch_label = LEADERSHIP_LABEL if best["target"] == -1 else ROLES[best["target"]]["label"]
    switch = _top_gaps(best["gap"])
    switch_cell = (
        f'<div><h4>Сменить сферу → {switch_label}</h4>'
        f'<ul class="growth">{_li(switch)}</ul></div>'
    )

    return f"""
<div class="role card">
  <div class="role-head"><h3>{role['label']}</h3>
    <span class="titles">какие компетенции растить</span></div>
  <div class="grid2" style="grid-template-columns:1fr 1fr 1fr">
    <div><h4>Расти в своей роли → C-level</h4><ul class="growth">{_li(excel)}</ul></div>
    {switch_cell}
    <div><h4>Стать руководителем (→ {LEADERSHIP_LABEL})</h4>
      <ul class="growth">{_li(lead)}</ul></div>
  </div>
</div>
"""


def _team_guide_html(axis_labels: dict[str, str]) -> str:
    """Team-building guide organised by TRL stage: from a few people at early
    stages to a full team at production stages."""
    rows = ""
    for s in TRL_STAGES:
        members = STAGE_TEAMS[s["id"]]
        total = sum(c for _, _, c in members)
        req = STAGE_REQUIREMENTS[s["id"]]["axes"]
        top2 = sorted(req, key=req.get, reverse=True)[:2]
        top2_names = " + ".join(axis_labels[a] for a in top2)
        rows += (
            f'<tr><td><b>УГТ {s["trl"]}</b></td><td>{s["name"]}</td>'
            f'<td>{total}</td><td>{top2_names}</td></tr>'
        )
    summary = (
        '<table class="rolemap"><tr><th>Стадия</th><th>Направление</th>'
        '<th>Команда</th><th>Ключевые компетенции</th></tr>'
        f'{rows}</table>'
    )

    cards = ""
    for s in TRL_STAGES:
        members = STAGE_TEAMS[s["id"]]
        total = sum(c for _, _, c in members)
        chips = "".join(
            f'<span class="trl-chip">{ROLES[rid]["label"]} · {GRADES[gid]["name"]}'
            f'{" ×" + str(cnt) if cnt > 1 else ""}</span>'
            for rid, gid, cnt in members
        )
        leaders = [m for m in members if GRADES[m[1]]["id"] >= 3]
        lead = " · ".join(
            f"{ROLES[rid]['label']} ({GRADES[gid]['name']})" for rid, gid, _ in leaders
        )
        cards += (
            f'<div class="card" style="margin-bottom:16px">'
            f'<h3 style="margin:0 0 6px">УГТ {s["trl"]} · {s["name"]} '
            f'<span class="titles">— {total} чел.</span></h3>'
            f'<div class="trl-chips" style="margin-bottom:8px">{chips}</div>'
            f'<p class="note">Минимальный старт — {lead}.</p>'
            f'</div>'
        )
    return f"""
<div class="card" style="margin-bottom:20px">{summary}</div>
{cards}
"""


def _references_html() -> str:
    items = "".join(
        f'<li><a href="{url}" target="_blank" rel="noopener">{name}</a> — {note}</li>'
        for name, url, note in REFERENCES
    )
    paras = "".join(f"<p>{p}</p>" for p in RATIONALE)
    return (
        "<h3>Почему отчёт устроен так</h3>" + paras +
        "<h3>Другие модели компетенций (для сверки)</h3>"
        f"<ul class=\"refs\">{items}</ul>"
    )


def _trl_badge(role_id: int) -> str:
    m = TRL_MAP[role_id]
    if m["transversal"]:
        return "сквозная · УГТ 0–9"
    s = TRL_STAGES[m["stage_id"]]
    return f"УГТ {s['trl']} · {m['family']}"


def _trl_map_html() -> str:
    stage_roles: dict[int, list[str]] = {s["id"]: [] for s in TRL_STAGES}
    transversal_roles: list[str] = []
    for rid, m in TRL_MAP.items():
        label = ROLES[rid]["label"]
        if m["transversal"]:
            transversal_roles.append(label)
        elif m["stage_id"] is not None:
            span = m["span"]
            stage_roles[m["stage_id"]].append(f"{label} ({span[0]}–{span[1]})")

    segments = ""
    for s in TRL_STAGES:
        chips = "".join(
            f'<span class="trl-chip">{c}</span>' for c in stage_roles[s["id"]]
        )
        segments += (
            f'<div class="trl-seg" style="flex:{s["w"]}">'
            f'<div class="trl-band">УГТ {s["trl"]}</div>'
            f'<div class="trl-name">{s["name"]}</div>'
            f'<div class="trl-family">{s["family"]}</div>'
            f'<div class="trl-chips">{chips}</div>'
            f'</div>'
        )

    artifacts = ""
    for family, arts in ARTIFACTS.items():
        items = "".join(f"<li>{a}</li>" for a in arts)
        artifacts += (
            f'<div class="trl-art"><h4>{family}</h4>'
            f'<ul class="growth">{items}</ul></div>'
        )

    transv = " · ".join(transversal_roles)
    return f"""
<div class="card">
  <div class="trl-bar">{segments}</div>
  <div class="trl-transv"><b>Сквозные роли (работают на всех стадиях):</b> {transv}</div>
  <h3 style="margin-top:20px">Доказательные результаты (артефакты) по ролям</h3>
  <div class="trl-arts">{artifacts}</div>
  <p class="note">УГТ (уровень готовности технологий, ГОСТ Р 71726-2024) — зрелость
  технологии; грейд роли (Junior → C-level) — зрелость специалиста. Оси независимы:
  младший специалист может работать на зрелой производственной линии (УГТ 8–9), а
  C-level — на фундаментальной стадии (УГТ 0–3).</p>
</div>
"""


def _role_map_html() -> str:
    """Render the role map (УГТ × грейд) as a matrix of проектные функции."""
    header = "".join(
        f'<th>УГТ {s["trl"]}<br><span class="th-sub">{s["name"]}</span>'
        f'<br><span class="th-fam">{s["family"]}</span></th>'
        for s in TRL_STAGES
    )

    rows = ""
    for g in GRADES:
        cells = ""
        for s in TRL_STAGES:
            cell = next(
                (m["title"] for m in ROLE_MAP
                 if m["stage"] == s["id"] and m["grade"] == g["id"]),
                "",
            )
            cells += f'<td>{cell}</td>'
        rows += (
            f'<tr><th>{g["name"]}<br>'
            f'<span class="th-sub">{g["dreyfus"]}</span></th>{cells}</tr>'
        )

    # transversal roles ladder: грейд progression for roles that span all stages
    transv_rows = ""
    for rid, ladder in TRANSVERSAL_LADDER.items():
        cells = "".join(f'<td>{ladder[g["id"]]}</td>' for g in GRADES)
        transv_rows += f'<tr><th>{ROLES[rid]["label"]}</th>{cells}</tr>'

    transv_header = "".join(
        f'<th>{g["name"]}</th>' for g in GRADES
    )

    return f"""
<div class="card">
  <h3 style="margin-top:0">Карта ролей: проектные функции на пересечении УГТ и грейда</h3>
  <div class="table-scroll"><table class="rolemap">
    <tr><th>Грейд роли</th>{header}</tr>
    {rows}
  </table></div>
  <h3 style="margin-top:22px">Сквозные роли: лестница грейдов (работают на всех стадиях УГТ)</h3>
  <div class="table-scroll"><table class="rolemap">
    <tr><th>Сквозная роль</th>{transv_header}</tr>
    {transv_rows}
  </table></div>
  <p class="note">Грейд роли — уровень сформированности компетенций (модель Дрейфуса):
  от Исполнителя (новичок) до Создателя (эксперт). Экспертные грейды — это PI (ведущий
  исследователь), PE (ведущий инженер) и PP (лидер продуктового направления). Семь
  карьерных уровней свёрнуты в пять грейдов — разрыв компетенций между уровнями
  значительный. Сквозные роли (биоинформатик, аналитик данных, менеджер лаборатории)
  не привязаны к одной стадии — они растут по грейдам и нужны на каждой стадии УГТ.</p>
</div>
"""


def _stage_role_model(stage_id: int, axis_labels: dict[str, str]) -> dict[str, Any]:
    req = STAGE_REQUIREMENTS[stage_id]["axes"]
    return {
        "role_id": stage_id,
        "role_label": f"УГТ {TRL_STAGES[stage_id]['trl']} · {TRL_STAGES[stage_id]['name']}",
        "top_job_titles": [],
        "member_count": sum(c for _, _, c in STAGE_TEAMS[stage_id]),
        "characteristics": [
            {"characteristic_id": a, "label_ru": axis_labels[a],
             "proficiency": req[a], "top_skills": []}
            for a in AXIS_IDS
        ],
        "competency_levels": [],
    }


def _stage_coverage(stage_id: int) -> dict[str, tuple[float, float]]:
    """Compute (team supply, stage requirement) per axis. Supply = max over
    team members of the role's value at that member's грейд."""
    req = STAGE_REQUIREMENTS[stage_id]["axes"]
    supply = dict.fromkeys(AXIS_IDS, 0.0)
    for rid, gid, _count in STAGE_TEAMS[stage_id]:
        t = GRADES[gid]["t"]
        role = ROLES[rid]
        for a in AXIS_IDS:
            supply[a] = max(supply[a], _grow(role["junior"][a], role["clevel"][a], t, a))
    return {a: (supply[a], req[a]) for a in AXIS_IDS}


def _stage_card_html(s: dict, axis_labels: dict[str, str], radar: str | None) -> str:
    sid = s["id"]
    team = "".join(
        f'<span class="trl-chip">{ROLES[rid]["label"]} · {GRADES[gid]["name"]}'
        f'{" ×" + str(cnt) if cnt > 1 else ""}</span>'
        for rid, gid, cnt in STAGE_TEAMS[sid]
    )
    cov = _stage_coverage(sid)
    cov_rows = ""
    for a in AXIS_IDS:
        sup, rq = cov[a]
        ok = sup >= rq - 0.01
        mark = '<span class="cok">✓</span>' if ok else '<span class="cno">дефицит</span>'
        cov_rows += (
            f'<tr><td>{axis_labels[a]}</td><td>{rq:.1f}</td>'
            f'<td>{sup:.1f}</td><td>{mark}</td></tr>'
        )
    diagram = radar or '<p class="note">нет данных</p>'
    return f"""
<div class="stage-card">
  <div class="grid2">
    <div class="cell">
      <h4>Требуемый профиль компетенций стадии</h4>
      <div class="frame">{diagram}</div>
    </div>
    <div class="cell">
      <h4>Команда (роль · грейд)</h4>
      <div class="trl-chips" style="margin-bottom:12px">{team}</div>
      <h4>Покрытие компетенций: требование vs команда</h4>
      <table class="rolemap">
        <tr><th>Компетенция</th><th>Требуется</th><th>Команда</th><th></th></tr>
        {cov_rows}
      </table>
    </div>
  </div>
</div>
"""


def _merged_model_html(axis_labels: dict[str, str],
                       charts: dict[str, str | None],
                       stage_radars: dict[int, str]) -> str:
    """Merged model: УГТ stages × roles, each role at its stage with its full
    competency model, plus transversal roles."""
    residents: dict[int, list[int]] = {s["id"]: [] for s in TRL_STAGES}
    transversal: list[int] = []
    for rid, m in TRL_MAP.items():
        if m["transversal"]:
            transversal.append(rid)
        elif m["stage_id"] is not None:
            residents[m["stage_id"]].append(rid)

    parts: list[str] = []
    for s in TRL_STAGES:
        sid = s["id"]
        parts.append(
            f'<h3 style="margin-top:30px">УГТ {s["trl"]} · {s["name"]} · '
            f'<span style="color:var(--accent)">{s["family"]}</span></h3>'
        )
        parts.append(_stage_card_html(s, axis_labels, stage_radars[sid]))
        for rid in residents[sid]:
            parts.append(_role_html(ROLES[rid], axis_labels, charts))

    parts.append('<h3 style="margin-top:30px">Сквозные роли (работают на всех стадиях УГТ)</h3>')
    for rid in transversal:
        parts.append(_role_html(ROLES[rid], axis_labels, charts))

    return "".join(parts)



def _soft_chart(role: MockRole, rid: int) -> str | None:
    """Interactive soft-competence spider, or ``None`` when the role has none."""
    soft = role.get("soft_scores") or []
    if not soft:
        return None
    model = {
        "role_label": role["label"],
        "member_count": role["member_count"],
        "top_job_titles": role.get("top_titles"),
        "soft_competences": soft,
    }
    order = [s["soft_id"] for s in soft]
    return render_role_soft_spider_html(model, order, div_id=f"soft_{rid}")


def _experience_chart(role: MockRole, rid: int) -> str | None:
    """Interactive experience histogram, or ``None`` when the role has none."""
    exp = role.get("experience") or {}
    bins = exp.get("bins") or []
    n_unspecified = int(exp.get("n_not_specified") or 0)
    if not bins and not n_unspecified:
        return None
    labelled = [(str(b["range"]), int(b["count"])) for b in bins]
    return render_role_experience_html(
        role["label"], labelled, exp.get("median"), n_unspecified,
        div_id=f"experience_{rid}",
    )


def _skills_chart(role: MockRole, axis_labels: dict[str, str], rid: int) -> str | None:
    """Interactive top-skills chart, or ``None`` when the role has none."""
    detail = role.get("skills_detail") or []
    if not detail:
        return None
    return render_role_skills_html(
        role["label"], detail, axis_labels, div_id=f"skills_{rid}",
    )


def _build_html(models: list[dict[str, Any]], roles: list[MockRole],
                axis_labels: dict[str, str]) -> str:
    charts: dict[str, str | None] = {}
    for r in roles:
        rid = r["role_id"]
        model = next((m for m in models if m["role_id"] == rid), None)
        charts[f"spider_{rid}"] = (
            render_role_spider_html(model, AXIS_IDS, div_id=f"spider_{rid}")
            if model is not None else None
        )
        charts[f"soft_{rid}"] = _soft_chart(r, rid)
        charts[f"experience_{rid}"] = _experience_chart(r, rid)
        charts[f"skills_{rid}"] = _skills_chart(r, axis_labels, rid)
    stage_radars = {
        s["id"]: render_role_spider_html(
            _stage_role_model(s["id"], axis_labels), AXIS_IDS,
            div_id=f"stage_radar_{s['id']}",
        )
        for s in TRL_STAGES
    }
    comparison = render_comparison_html(
        models, AXIS_IDS, include_plotlyjs="cdn", div_id="comparison")
    summary = render_summary_html(
        [r["label"] for r in roles],
        [r["member_count"] for r in roles],
        [r["skill_count"] for r in roles],
        div_id="summary",
    )
    leadership = render_leadership_gap_html(
        roles, _leadership_edges(), axis_labels, div_id="leadership_gap")
    skills_by_role = {r["role_id"]: str(r["skill_count"]) for r in roles}
    growth_sections = "".join(_growth_html(r, axis_labels) for r in roles)
    constructor = _role_constructor_html(axis_labels)
    team_constructor = _team_constructor_html(axis_labels)
    team_guide = _team_guide_html(axis_labels)
    transition_net = _transition_network_html(axis_labels, include_js=False)
    trl_map = _trl_map_html()
    role_map = _role_map_html()
    merged_model = _merged_model_html(axis_labels, charts, stage_radars)
    return f"""<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>КРМ — Компетентностно-Ролевая Модель</title>
<style>{_CSS}</style></head>
<body>
<header>
  <div class="wrap">
    <h1>Компетентностно-Ролевая Модель STEM-рынка труда</h1>
    <p class="sub">Эталон итогового отчёта · данные: иллюстративный пример</p>
    <div class="badges">
      <span class="badge">Домен: биология</span>
      <span class="badge">6 ролей</span>
      <span class="badge">7 компетенций</span>
      <span class="badge">версия 0.2.0</span>
    </div>
  </div>
</header>
<div class="wrap">
  <section>
    <h2>Сводка</h2>
    <div class="kpis">{_kpi_html()}</div>
  </section>

  <section>
    <h2>Валидационные метрики</h2>
    <div class="card">{_metrics_html()}</div>
    <div class="card" style="margin-top:18px">{_metric_explain_html()}</div>
  </section>

  <section>
    <h2>Обзор ролей</h2>
    <div class="card">{_roles_table_html(models, skills_by_role)}</div>
    <div class="card" style="margin-top:18px">
      <h3>Сравнение профилей ролей</h3>
      {comparison}
    </div>
  </section>

  <section>
    <h2>Модель КРМ: роли на стадиях УГТ</h2>
    <p class="lead">Роли помещены на свою стадию технологической зрелости (УГТ). Для
    каждой стадии показан требуемый профиль компетенций, команда, которая его
    покрывает, и полные модели компетенций входящих ролей. Сквозные роли работают
    на всех стадиях.</p>
    {trl_map}
    {merged_model}
    <h3 style="margin-top:30px">Карта ролей: проектные функции (УГТ × грейд)</h3>
    {role_map}
  </section>

  <section>
    <h2>Карта перехода между ролями</h2>
    {transition_net}
    <p class="note">Переход возможен, если роли имеют хотя бы одну общую доминантную
    компетенцию и близкие профили (косинусное сходство C-level профилей ≥
    {TRANSITION_MIN_SIM:.1f} при ограниченном разрыве компетенций). На карте — по
    {TRANSITION_MAX_DEGREE} ближайших перехода для каждой роли. Руководство — не отдельная
    базовая роль, а цель: её достигают, развивая «{axis_labels['management']}» и
    «{axis_labels['domain_knowledge']}» до уровня 4.0.</p>
    <div class="card" style="margin-top:18px">
      {leadership}
    </div>
  </section>

  <section>
    <h2>Конструктор ролей</h2>
    <p class="lead">Попробуйте собрать собственный профиль: задайте уровень каждой
    компетенции и посмотрите, к какой роли и карьерному уровню он ближе всего.</p>
    {constructor}
  </section>

  <section>
    <h2>Рост с уровня Junior</h2>
    <p>Мы смоделировали базовые (junior) роли. Ниже — какие компетенции нужно
    развивать, чтобы расти в своей роли, сменить сферу или стать руководителем.</p>
    {growth_sections}
  </section>

  <section>
    <h2>Гайд для бизнеса: как собрать команду под стадию УГТ</h2>
    <p class="lead">Чем выше стадия технологической зрелости, тем больше команда: от
    одного–двух человек на фундаментальных исследованиях до полной производственной
    команды на внедрении. Ниже — состав команды для каждой стадии УГТ.</p>
    {team_guide}
    <h3 style="margin-top:24px">Проверьте свою команду</h3>
    <p class="note">Добавьте роли и посмотрите, какие компетенции закрыты, а каких не хватает.</p>
    {team_constructor}
  </section>

  <section>
    <h2>Статистика</h2>
    <div class="card">{summary}</div>
  </section>

  <section>
    <h2>Справочные материалы и обоснование</h2>
    <div class="card">{_references_html()}</div>
    <p class="note">Данные иллюстративны. Каждая роль определена доминантной парой
    компетенций; по мере роста от Junior к C-level все компетенции только растут
    (не снижаются), а доминантная пара сохраняет лидерство на каждом уровне.</p>
  </section>
</div>
<footer>
  Компетентностно-Ролевая Модель (КРМ) · иллюстративный отчёт · версия 0.2.0
</footer>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = Config()
    axis_labels = config.characteristic_labels_ru
    out_dir = config.reports_dir / "etalon"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [_to_role_model(r, axis_labels) for r in ROLES]

    html = _build_html(models, ROLES, axis_labels)
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"Эталон отчёт → {report_path}")


if __name__ == "__main__":
    main()
