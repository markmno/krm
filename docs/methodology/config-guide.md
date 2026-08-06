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

axes:
  - id: experimental
    label_ru: Экспериментальный опыт
    hypothesis: "This skill involves hands-on laboratory or field experimental work"
  - id: domain_knowledge
    label_ru: Предметные знания
    hypothesis: "This skill involves theoretical understanding of a scientific domain"
  - id: management
    label_ru: Управление и коммуникации
    hypothesis: "This skill involves project management, coordination, or communication"
  - id: literature
    label_ru: Научная литература и документация
    hypothesis: "This skill involves reading, writing, or organizing scientific documents"
  - id: data_analysis
    label_ru: Анализ данных и статистика
    hypothesis: "This skill involves statistical analysis, data processing, or quantitative research"
  - id: computational
    label_ru: Вычислительные методы
    hypothesis: "This skill involves computational modeling, simulation, or algorithm development"

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

### axes

6 фиксированных осей компетенций. Каждая содержит:
- `id` — machine-readable идентификатор
- `label_ru` — русское название для spider charts
- `hypothesis` — английская NLI-гипотеза

## Как изменить ось

Добавить седьмую ось "Преподавание":
```yaml
axes:
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
- Все axes имеют уникальные id
