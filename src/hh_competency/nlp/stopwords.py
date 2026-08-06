"""Three-tier vacancy text filter: stopwords, skill buzzwords, and IT exclusion.

Tier 1: General Russian stopwords (common nouns like место, время, год).
Tier 2: Skill buzzwords (job-posting jargon like опыт, вакансия, разработка).
         These are excluded from skill extraction — they appear in every posting.
Tier 3: IT exclusion keywords — technology-specific terms that identify pure
         IT vacancies. Whole vacancy is excluded if any keyword is found in
         title or description. Python, C++, MATLAB, R are NOT in this list
         (they are scientific computing tools).

Filtering is case-insensitive and lemma-based.
"""

from __future__ import annotations


class StopwordFilter:
    """Filters stopwords and identifies IT-only vacancies."""

    def __init__(self) -> None:
        self._general: frozenset[str] = _load_general_stopwords()
        self._skill_buzzwords: frozenset[str] = _load_skill_buzzwords()
        self._it_keywords: frozenset[str] = _load_it_exclusion_keywords()

    def is_stopword(self, lemma: str) -> bool:
        """Check if lemma is a stopword (general OR skill buzzword).

        IT keywords are NOT stopwords — they are vacancy-level markers.
        """
        lower = lemma.lower()
        return lower in self._general or lower in self._skill_buzzwords

    def is_it_vacancy(self, name: str, description: str | None = None) -> bool:
        """Check if a vacancy is pure IT — should be excluded from science analysis.

        Returns True if the vacancy title or description contains any
        IT-specific technology keyword. Python, C++, MATLAB, R are
        NOT considered IT (scientific computing tools).
        """
        text = (name + " " + (description or "")).lower()
        return any(kw.lower() in text for kw in self._it_keywords)

    @property
    def general_count(self) -> int:
        return len(self._general)

    @property
    def skill_buzzword_count(self) -> int:
        return len(self._skill_buzzwords)

    @property
    def it_keyword_count(self) -> int:
        return len(self._it_keywords)

    @property
    def total_count(self) -> int:
        return self.general_count + self.skill_buzzword_count + self.it_keyword_count


def _load_general_stopwords() -> frozenset[str]:
    """Standard Russian stopwords — common nouns with no domain signal."""
    return frozenset([
        "место", "время", "год", "человек", "дело", "жизнь", "день",
        "рука", "раз", "работа", "слово", "лицо", "друг", "глаз",
        "вопрос", "сторона", "страна", "мир", "случай", "голова",
        "ребенок", "сила", "конец", "вид", "система", "часть",
        "город", "отношение", "женщина", "деньги", "земля",
        "машина", "вода", "отец", "проблема", "право",
        "дверь", "образ", "закон", "война", "бог",
        "голос", "тысяча", "книга", "возможность", "результат",
        "ночь", "стол", "имя", "область", "статья",
        "число", "компания", "народ", "жена", "группа",
        "развитие", "процесс", "суд", "условие", "средство",
        "начало", "свет", "пора", "путь", "душа",
        "уровень", "форма", "связь", "минута", "улица",
        "вечер", "качество", "мысль", "дорога", "мать",
        "действие", "месяц", "государство", "язык", "любовь",
        "взгляд", "мама", "век", "школа", "цель",
        "общество", "деятельность", "организация", "президент", "комната",
        "порядок", "момент", "письмо", "утро", "помощь",
        "ситуация", "роль", "рубль", "смысл", "состояние",
        "квартира", "внимание", "смерть", "выход", "правда",
        "образование", "чувство", "правило", "сомнение", "движение",
        "материал", "основа", "муж", "метр", "решение",
        "сон", "оценка", "размер", "представитель", "пример",
        "интерес", "стекло", "ход", "история", "партия",
        "счет", "участок", "тело", "очко", "период",
        "документ", "центр", "спина", "гость", "брат",
        "плечо", "причина", "палец", "народ", "позиция",
        "защита", "механизм", "ряд", "объект", "способ",
        "край", "лес", "искусство", "обед", "кресло",
        "свобода", "удар", "фон", "середина", "пара",
        "масло", "зона", "мужчина", "цвет", "отдел",
        "сентябрь", "сентябрь", "бумага", "июль", "институт",
        "студент", "мечта", "волос", "деревня", "факт",
        "характер", "мнение", "вещь", "площадь", "очередь",
        "руководитель", "повод", "слеза", "разговор", "пространство",
        "течение", "одежда", "врач", "камень", "собака",
        "расход", "память", "кухня", "река", "страх",
        "сестра", "стих", "фильм", "пол", "ширина",
        "спектакль", "карман", "организм", "огонь", "технология",
        "высота", "прием", "дерево", "половина", "июнь",
        "поток", "праздник", "использование", "май", "срок",
    ])


def _load_skill_buzzwords() -> frozenset[str]:
    """Words that appear in every job posting and pollute skill extraction.

    These are NOT domain-specific skills — they are generic job-posting
    filler words, bureaucratic terms, and concepts too broad to be useful
    for competency axis discovery.
    """
    return frozenset([
        # Job-posting furniture (HR jargon)
        "опыт", "навык", "требование", "обязанность", "условие",
        "компания", "вакансия", "кандидат", "резюме", "знание",
        "образование", "сотрудник", "специалист", "возможность",
        "оплата", "преимущество", "график", "офис", "требоваться",
        "приветствовать", "информация", "контакт", "телефон",
        "почта", "адрес", "регион", "метро", "пожелание",
        "анкета", "собеседование", "портфолио", "испытательный",
        "полный", "неполный", "наличие", "отсутствие",
        "обучение", "заработный", "плата", "оклад", "ставка",
        "выплата", "премия", "бонус", "соцпакет", "социальный",
        "пакет", "оформление", "трудовой", "договор", "кодекс",

        # Generic action nouns (appear in every posting)
        "разработка", "данные", "анализ", "тестирование", "понимание",
        "модель", "проведение", "умение", "задача", "участие",
        "подготовка", "создание", "контроль", "поддержка", "взаимодействие",
        "реализация", "выполнение", "формирование", "документация", "ведение",
        "составление", "соответствие", "рамка", "желание", "обеспечение",
        "процесс", "результат", "качество", "эффективность", "срок",
        "план", "стратегия", "оптимизация", "автоматизация", "внедрение",
        "сопровождение", "администрирование", "мониторинг", "согласование",

        # Cross-domain noise (appear in IT and science, zero role discrimination)
        "алгоритм", "продукт", "команда", "проект", "стек",
        "рекомендация", "вывод", "фреймворк", "инфраструктура", "принцип",
        "платформа", "сервис", "решение", "приложение", "интерфейс",
        "бизнес", "клиент", "пользователь", "рынок", "заказчик",
        "инструмент", "подход", "методология", "практика", "функционал",
        "компонент", "модуль", "архитектура", "технология",
        "эффективность", "надёжность", "производительность", "масштабируемость",
        "гибкость", "стабильность", "скорость", "направление", "интеграция",
    ])


def _load_it_exclusion_keywords() -> frozenset[str]:
    """Technology-specific keywords that indicate a pure IT vacancy.

    If any of these keywords appear in vacancy title or description,
    the vacancy should be excluded from scientific competency analysis.

    Deliberately EXCLUDED (scientific computing tools):
    - python, c++, cpp, matlab, r
    """
    return frozenset([
        # Programming languages (excluding Python, C++, MATLAB, R)
        "java", "javascript", "typescript", "c#", "csharp",
        "kotlin", "swift", "ruby", "php", "golang", "go",
        "scala", "perl", "lua", "dart", "groovy",

        # Frontend frameworks
        "react", "angular", "vue", "svelte", "nextjs", "next.js",
        "nuxtjs", "bootstrap", "jquery",

        # CMS
        "wordpress", "drupal",

        # Backend frameworks
        "django", "flask", "spring", "laravel", "expressjs",
        "nestjs",

        # Runtime platforms
        "nodejs", "node.js", "dotnet",

        # Mobile & game
        "android", "ios", "flutter", "unity", "unreal",
        "gamedev", "game dev",

        # Role labels
        "frontend", "front-end", "backend", "back-end",
        "fullstack", "full-stack", "devops",

        # DevOps & cloud
        "docker", "kubernetes", "k8s", "terraform", "jenkins",
        "ansible", "aws", "azure",
        "gitlab", "bitbucket", "github actions",

        # Infrastructure
        "rest api", "graphql", "microservice",

        # Databases (as dedicated DB admin/RDBMS roles)
        "mongodb", "postgresql", "mysql", "redis",

        # Tools
        "jira", "confluence", "trello", "slack", "figma",

        # Methodologies
        "agile", "scrum", "kanban",
    ])
