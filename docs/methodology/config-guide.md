# Руководство по конфигурации

## config.yaml — единый источник параметров

```yaml
pipeline:
  raw_dir: data/raw
  output_dir: data
  models_dir: models
  reports_dir: reports

collection:
  date_from: "2010-01-01"
  keywords: [физик, химик, биолог, эколог, научный сотрудник]
  professional_roles: [79]   # R&D
  categories: [25]           # Science & Education
  exclude_roles: [96]        # IT, Internet, Telecom
  rate_limit_rps: 2

classification:
  model: facebook/bart-large-mnli
  stem_threshold: 0.6
  it_threshold: 0.6
  interdisciplinary_threshold: 0.5

roles:
  embedding_model: intfloat/multilingual-e5-large-instruct
  umap_n_components: 16
  hdbscan_min_cluster_size: 15
  hdbscan_cluster_selection_epsilon: 0.25

skills:
  esco_version: v1.2.1
  esco_path: data/esco_skills_en.csv
  fuzzy_threshold: 0.7
  tfidf_max_features: 5000

characteristics:
  model: facebook/bart-large-mnli
  device: cuda
  batch_size: 32
  confidence_threshold: 0.3
  hypotheses:
    - id: domain_knowledge
      label_ru: Доменная база
      hypothesis: "This skill requires deep theoretical or applied knowledge in a specific scientific or technical domain."
    - id: experimental
      label_ru: Эксперимент
      hypothesis: "This skill involves hands-on laboratory or field experimental work with equipment, organisms, or materials."
    - id: data_analysis
      label_ru: Анализ данных
      hypothesis: "This skill involves analyzing data, performing statistical tests, or applying quantitative methods."
    - id: computational
      label_ru: Вычислительные методы
      hypothesis: "This skill involves computational modeling, programming, algorithm development, or numerical methods."
    - id: professional_texts
      label_ru: Профессиональные тексты
      hypothesis: "This skill involves writing, reading, or working with professional and scientific texts, documentation, publications, and reports."
    - id: t_profile
      label_ru: T-профиль
      hypothesis: "This skill reflects broad erudition and interdisciplinary breadth — knowledge that spans multiple STEM fields beyond a single specialty."
      skill_category_triggers: []
    - id: management
      label_ru: Управление
      hypothesis: "This skill involves managing teams, projects, budgets, or organizational processes."

soft_competences:
  model: facebook/bart-large-mnli
  device: cuda
  batch_size: 32
  confidence_threshold: 0.3
  min_descriptions: 5
  hypotheses:
    - id: thinking
      label_ru: Мышление
      hypothesis: "This role requires critical thinking, analytical reasoning, problem-solving, or intellectual creativity."
    - id: teamwork
      label_ru: Командное Взаимодействие
      hypothesis: "This role requires teamwork, collaboration, coordinating with colleagues, or interdisciplinary cooperation."
    - id: leadership
      label_ru: Ответственность и Лидерство
      hypothesis: "This role requires taking responsibility, leading people or projects, decision-making, or mentoring others."
    - id: professional_culture
      label_ru: Профессиональная культура
      hypothesis: "This role requires professional ethics, scientific integrity, adherence to standards and norms, or responsible professional conduct."
  baseline:
    Техник: {thinking: 2, teamwork: 2, leadership: 1, professional_culture: 2}
    Исследователь: {thinking: 4, teamwork: 3, leadership: 2, professional_culture: 3}
    Методолог: {thinking: 4, teamwork: 3, leadership: 2, professional_culture: 3}
    Инженер: {thinking: 3, teamwork: 2, leadership: 2, professional_culture: 2}
    Ведущий специалист: {thinking: 4, teamwork: 4, leadership: 4, professional_culture: 4}
    Гибрид: {thinking: 4, teamwork: 3, leadership: 3, professional_culture: 3}
  archetypes: [Техник, Исследователь, Методолог, Инженер, Ведущий специалист, Гибрид]

experience:
  bins: [0, 1, 3, 5, 10]   # histogram edges; last bin is 10+

skills_display:
  top_n: 15

testing:
  classification_sample_size: 200
  skill_precision_at_k: 10
  expert_validation_roles: 3
  bootstrap_iterations: 100
```

## Описание параметров

### pipeline

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `raw_dir` | `data/raw` | Где хранятся сырые HH.ru ответы |
| `output_dir` | `data` | Где хранятся промежуточные Parquet |
| `models_dir` | `models` | Где сохраняются JSON-модели |
| `reports_dir` | `reports` | Где сохраняются spider charts и отчёты |

### collection

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `date_from` | `"2010-01-01"` | Начальная дата сбора вакансий |
| `keywords` | `[физик, химик, ...]` | Ключевые слова поиска на HH.ru |
| `professional_roles` | `[79]` | ID роли на HH.ru (79 = R&D) |
| `categories` | `[25]` | ID категории (25 = Наука/Образование) |
| `exclude_roles` | `[96]` | Исключаемые роли (96 = ИТ/Телеком) |
| `rate_limit_rps` | `2` | Максимум запросов в секунду к HH.ru |

### classification

Пороги для многоуровневого классификатора STEM/IT:

| Параметр | Описание |
|----------|----------|
| `stem_threshold` | Минимальный entailment STEM-гипотезы |
| `it_threshold` | Минимальный entailment IT-гипотезы |
| `interdisciplinary_threshold` | Порог для междисциплинарных (STEM+IT > threshold) |

### roles

| Параметр | Описание | Рекомендация |
|----------|----------|-------------|
| `embedding_model` | Модель эмбеддингов | `multilingual-e5-large-instruct` (1024-dim) |
| `umap_n_components` | Размерность после UMAP | 16 (trade-off сжатия и сохранения структуры) |
| `hdbscan_min_cluster_size` | Минимальный размер кластера | 15 (меньше — больше шумовых ролей) |
| `hdbscan_cluster_selection_epsilon` | Чувствительность кластеризации | 0.25 (выше — меньше кластеров) |

### skills

| Параметр | Описание |
|----------|----------|
| `esco_version` | Версия ESCO taxonomy |
| `esco_path` | Путь к CSV с ESCO навыками |
| `fuzzy_threshold` | Cosine similarity > threshold для fuzzy match |
| `tfidf_max_features` | Максимум признаков для TF-IDF |

### characteristics

7 фиксированных hard-осей компетенций. Каждая содержит:
- `id` — machine-readable идентификатор
- `label_ru` — русское название для диаграмм
- `hypothesis` — английская NLI-гипотеза
- `skill_category_triggers` — (опционально) категории навыков для linking

Порядок осей значим: это порядок спиц на spider-диаграмме.

### soft_competences

4 soft-компетенции + per-archetype baseline:
- `hypotheses` — 4 оси: Мышление, Командное Взаимодействие, Ответственность и Лидерство, Профессиональная культура
- `baseline` — fallback-значения (шкала 1–5) на ось для каждого из 6 архетипов; используется, когда NLI-сигнал не проходит порог `confidence_threshold` / `min_descriptions`
- `archetypes` — список из 6 архетипов

### experience

`bins: [0, 1, 3, 5, 10]` — границы гистограммы опыта; последний бин — `10+`.

### skills_display

`top_n: 15` — сколько навыков показывать на диаграмме Skills.

## Как изменить ось

Добавить восьмую ось "Преподавание":
```yaml
characteristics:
  hypotheses:
    - id: teaching
      label_ru: Преподавание
      hypothesis: "This skill involves teaching, mentoring, curriculum development, or student supervision"
```

Удалить ось: просто удалить блок из YAML. NLI автоматически исключит её.

## Как настроить сбор под свой домен

Для сбора только химических вакансий:
```yaml
collection:
  keywords: [химик, химия, хроматография, спектроскопия, синтез]
```

Для сбора только физических:
```yaml
collection:
  keywords: [физик, физика, лазер, оптика, квантовый]
```

## Валидация конфигурации

При запуске пайплайн проверяет:
- Существование `esco_path`
- Корректность NLI-модели (загрузка без ошибок)
- Корректность embedding-модели
- Все `characteristics.hypotheses` имеют уникальные id
