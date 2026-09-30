"""Shared skill category classification rules.

Central source of truth for mapping skill lemmas to heuristic
competency categories. Used by both axis discovery and archetype
classification analysis modules.
"""

from __future__ import annotations

# Skill lemma → heuristic category mapping rules.
# Each rule: (trigger lemmas, category name in Russian, category name in English).
CATEGORY_RULES: list[tuple[list[str], str, str]] = [
    (
        [
            "питон", "python", "java", "c++", "cpp", "matlab", "r",
            "scala", "rust", "javascript", "typescript", "go", "golang",
            "программирование", "разработка", "программист", "developer",
            "бэкенд", "backend", "фронтенд", "frontend", "api", "rest",
            "граф", "git", "github", "gitlab", "docker", "kubernetes",
            "linux", "unix", "bash", "sql", "nosql", "база данных",
        ],
        "Программирование и IT",
        "Programming & IT",
    ),
    (
        [
            "xrd", "рентген", "sem", "tem", "afm", "спектрометр",
            "спектроскоп", "хроматограф", "масс-спектр", "микроскоп",
            "лабораторный", "оборудование", "прибор", "установка",
            "калибровка", "детектор", "сенсор", "лазерный",
            "вакуумный", "оптический", "измерение", "эксперимент",
            "протокол", "методика", "метод", "лаборант",
        ],
        "Экспериментальные методы и оборудование",
        "Experimental Methods & Equipment",
    ),
    (
        [
            "моделирование", "симуляция", "численный", "вычислительный",
            "алгоритм", "dft", "молекулярный", "динамика",
            "конечный элемент", "cfd", "hpc", "кластер",
            "параллельный", "gpu", "cuda", "квантовый", "расчёт",
            "оптимизация", "метод монте", "нейросеть", "нейронный",
            "машинный обучение", "машинный", "обучение",
            "искусственный интеллект", "ai", "ml", "dl",
        ],
        "Вычислительное моделирование",
        "Computational Modeling",
    ),
    (
        [
            "статистика", "анализ данных", "регрессия", "anova",
            "тест", "критерий", "выборка", "распределение",
            "корреляция", "факторный анализ", "кластеризация",
            "визуализация", "график", "диаграмма",
            "база данных", "sql", "обработка", "pandas",
            "очистка", "предобработка", "парсинг",
        ],
        "Анализ данных",
        "Data Analysis",
    ),
    (
        [
            "статья", "публикация", "журнал", "рецензируемый",
            "обзор", "конференция", "доклад", "презентация",
            "отчёт", "документация", "грант", "заявка",
            "научный", "исследование", "патент", "интеллектуальный",
            "результат", "внедрение", "коммерциализация",
        ],
        "Научная коммуникация и публикации",
        "Scientific Communication & Publications",
    ),
    (
        [
            "управление", "руководитель", "проект", "команда",
            "бюджет", "план", "срок", "отчётность",
            "закупка", "тендер", "поставщик", "логистика",
            "риск", "качество", "сертификация", "аудит",
            "нормативный", "регламент", "гост", "снип",
            "охрана труда", "техника безопасности",
        ],
        "Управление проектами и нормативы",
        "Project Management & Compliance",
    ),
    (
        [
            "клетка", "культура", "проба", "образец",
            "полевой", "экспедиция", "мониторинг", "наблюдение",
            "вид", "популяция", "экосистема", "среда",
            "in vitro", "in vivo", "фермент", "белок",
            "днк", "рнк", "пцр", "секвенирование",
            "геном", "протеом", "метаболом",
        ],
        "Биологические и полевые методы",
        "Biological & Field Methods",
    ),
    (
        [
            "синтез", "реакция", "катализ", "реагент",
            "растворитель", "экстракция", "дистилляция",
            "технология", "процесс", "производство",
            "масштабирование", "пилотный", "промышленный",
            "сырьё", "продукт", "выход", "чистота",
            "контроль", "сертификат", "gmp", "glp",
        ],
        "Химический синтез и технология",
        "Chemical Synthesis & Technology",
    ),
    (
        [
            "физика", "механика", "квантовый", "термодинамика",
            "электродинамика", "оптика", "фотоника",
            "полупроводник", "плазма", "ядерный",
            "акустика", "гидродинамика", "аэродинамика",
            "материал", "сплав", "полимер", "композит",
            "свойство", "структура", "фаза", "диаграмма",
        ],
        "Фундаментальная физика и материаловедение",
        "Fundamental Physics & Materials Science",
    ),
]


def classify_skill_to_category(
    skill_lemma: str,
    rules: list[tuple[list[str], str, str]] | None = None,
) -> str | None:
    """Classify a skill lemma into a heuristic category name.

    Returns None if no category matches (skill becomes its own axis
    or groups with other unclassified skills).

    Args:
        skill_lemma: The skill lemma string (e.g. 'python', 'хроматограф').
        rules: Optional custom category rules. Defaults to CATEGORY_RULES.

    Returns:
        Category name in Russian, or None if unclassified.
    """
    if rules is None:
        rules = CATEGORY_RULES

    skill_lower = skill_lemma.lower().strip()
    for triggers, name_ru, _name_en in rules:
        for trigger in triggers:
            # Trigger can be one word or multi-word
            if trigger in skill_lower or skill_lower in trigger:
                return name_ru
    return None
