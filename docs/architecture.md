# Архитектура КРМ (Компетентностно-Ролевая Модель)

## Обзор

КРМ — 7-фазный пайплайн для построения компетентностно-ролевых моделей STEM-рынка
труда на основе данных HH.ru. Каждая фаза — изолированный Python-модуль, читающий
Parquet-файлы с предыдущей фазы и записывающий результаты для следующей.
Файловая система — интерфейс между фазами.

Основное отличие от hh-competency v0.1: роли обнаруживаются из **названий
должностей** (embeddings + UMAP + HDBSCAN), а не из кластеризации навыков.
Оси компетенций — **фиксированы** (6 характеристик исследователя), навыки
маппятся на оси через zero-shot NLI, а не regex-правила.

## 7-фазный пайплайн

```
┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│ PHASE 1  │───▶│   PHASE 2    │───▶│    PHASE 3    │───▶│   PHASE 4    │───▶│   PHASE 5    │───▶│    PHASE 6    │───▶│   PHASE 7   │
│ Collect  │    │  Classify    │    │   Cluster     │    │   Extract    │    │   Map to     │    │    Build      │    │  Validate   │
│ Vacancies│    │ STEM/IT/non  │    │  Job Titles   │    │   Skills     │    │  6 Axes      │    │ Competencies  │    │  & Export   │
└──────────┘    └──────────────┘    └───────────────┘    └──────────────┘    └──────────────┘    └───────────────┘    └─────────────┘
      │               │                   │                    │                   │                    │                   │
      ▼               ▼                   ▼                    ▼                   ▼                    ▼                   ▼
  raw/*.parquet  classified.parquet   roles.parquet       skills_per_role.parquet  axis_scores.parquet   models/*.json     reports/

config.yaml (единый источник параметров: модели, пороги, пути)
```

### Фаза 1: Сбор вакансий (`phase_1_collect.py`)

**Вход**: `config.yaml` (API-ключи, диапазон дат, параметры поиска)

**Выход**: `data/raw/` — Parquet-файлы с сырыми JSON-ответами HH.ru

**Подход**:
- `kirillzhosul/hhru` Python SDK с обработкой 429 (rate limit)
- Фильтр API: `professional_role=79` (R&D) AND категория 25 (Наука/Образование)
- Исключение: `professional_role=96` (ИТ/телеком) через параметр `exclude`
- Инкрементальный сбор: проверка `max(created_at)` в существующих данных
- Хранение: DuckDB для дедупликации, Parquet для межфазного обмена

**Ограничение HH.ru API**: максимум 2000 вакансий на поисковый запрос.
Для обхода — множественные поисковые запросы с разными ключевыми словами.

### Фаза 2: Классификация STEM/IT/non-STEM (`phase_2_classify.py`)

**Вход**: `data/raw/*.parquet`

**Выход**: `data/classified.parquet` (+ колонка `stem_category`)

**4-уровневый классификатор**:

| Уровень | Метод | Что классифицирует |
|---------|------|--------------------|
| Tier 1 | API rule: professional_role in {96, 123-126} | PURE_IT — отсекается сразу |
| Tier 2 | Keyword triage: [Pp]rogramm/[Rr]azrabotchik без STEM-терминов | PURE_IT |
| Tier 3 | Zero-shot NLI: `bart-large-mnli` с 3 гипотезами | Ambiguous (~15%) |
| Tier 4 | Interdisciplinary catch: STEM + IT гипотезы > 0.6 | INTERDISCIPLINARY — сохраняется |

**Категории**: `STEM_RESEARCH | PURE_IT | NON_STEM | INTERDISCIPLINARY`

**Метрика**: F1 ≥ 0.92 на размеченной выборке из 200 вакансий (Cohen's κ для inter-annotator).

### Фаза 3: Кластеризация названий должностей в роли (`phase_3_roles.py`)

**Вход**: `data/classified.parquet` (фильтр: `STEM_RESEARCH`)

**Выход**: `data/roles.parquet` (role_id, role_label, centroid_embedding, top_titles, member_count)

**Алгоритм**:
1. Эмбеддинги всех уникальных названий должностей через `intfloat/multilingual-e5-large-instruct`
2. Снижение размерности: UMAP (`n_components=16`, `metric=cosine`, `min_dist=0.0`)
3. Кластеризация: HDBSCAN (`min_cluster_size=15`, `cluster_selection_epsilon=0.25`)
4. Именование кластеров: топ-3 наиболее центральных названия должностей
5. Мягкая кластеризация: `all_points_membership_vectors_` для междисциплинарных ролей

**Метрики качества**:
- Silhouette score (цель > 0.4)
- Davies-Bouldin index
- Bootstrap ARI (80% подвыборки, 100 итераций)
- Экспертная валидация: 100 названий должностей → adjusted Rand index против ground truth

**Важно**: HDBSCAN помечает выбросы как кластер -1. Вакансии в этом кластере попадают
в "Miscellaneous STEM Researcher" и не форсируются в существующие кластеры.

### Фаза 4: Извлечение навыков на роль (`phase_4_skills.py`)

**Вход**: `data/roles.parquet` + `data/classified.parquet` (JOIN по vacancy_id → role_id)

**Выход**: `data/skills_per_role.parquet` (role_id, skill_canonical_name, esco_id, frequency, tfidf_weight)

**Подход**:
1. Аггрегация всех описаний вакансий на роль
2. Извлечение навыков: dictionary-based matching с ESCO v1.2.1 (13,939 навыков)
   + Russian Profstandart skills (40.008, 40.011)
   + fuzzy matching (порог косинусного сходства: 0.7)
3. Нормализация: русские термины → ESCO preferred label через multilingual-e5
4. TF-IDF взвешивание: `tfidf_weight = term_frequency_in_role × log(N_roles / role_frequency)`

**Метрика**: Precision@10 ≥ 0.85 (экспертная разметка топ-20 навыков для 5 ролей).

**Некаталогизированные навыки** (cosine < 0.7 с ESCO): сохраняются в
`uncatalogued_skills` и отправляются на ручную классификацию.

### Фаза 5: Маппинг навыков на 6 осей (`phase_5_axes.py`)

**Вход**: `data/skills_per_role.parquet`

**Выход**: `data/axis_scores.parquet` (role_id, axis_name, proficiency_1_to_5, top_contributing_skills)

**6 фиксированных осей** (заданы в `config.yaml`, не обнаруживаются из данных):

| # | Ось | NLI-гипотеза |
|---|-----|-------------|
| 1 | Экспериментальный опыт | "This skill involves hands-on laboratory or field experimental work" |
| 2 | Предметные знания | "This skill involves theoretical or conceptual understanding of a scientific domain" |
| 3 | Управление и коммуникации | "This skill involves project management, team coordination, or scientific communication" |
| 4 | Научная литература и документация | "This skill involves reading, writing, or organizing scientific documents" |
| 5 | Анализ данных и статистика | "This skill involves statistical analysis, data processing, or quantitative research" |
| 6 | Вычислительные методы | "This skill involves computational modeling, simulation, or algorithm development" |

**Алгоритм**:
1. Для каждого навыка: zero-shot NLI (`bart-large-mnli`) против 6 гипотез
2. Мягкое распределение: один навык может принадлежать нескольким осям
   (e.g., "молекулярная динамика" → 0.7 domain_knowledge, 0.6 computational)
3. Аггрегация на роль: `proficiency = Σ(skill.tfidf_weight × axis_score) / Σ(all_weights)`
4. Нормализация: шкала 1–5 (ознакомительный → экспертный)

**Валидация**: Spearman ρ ≥ 0.75 против экспертных оценок (3 роли × 6 осей = 18 суждений).

### Фаза 6: Построение компетентностных моделей и spider charts (`phase_6_model.py`)

**Вход**: `data/axis_scores.parquet` + `data/roles.parquet`

**Выход**: `models/{role_id}.json` + `reports/spider_charts/{role_id}.png`

**Компетентностная модель JSON**:
- Role label и топ-10 названий должностей
- 7-уровневая шкала (бакалавр → ГНС)
- Per-axis proficiency (1–5) + топ-3 навыка-драйвера
- Кросс-референсы на Profstandart 40.008 и 40.011

**Spider chart**: 6 осей, шкала 1–5, подписи на русском, matplotlib (публикационный PNG).

### Фаза 7: Валидация и отчёты (`phase_7_validate.py`)

**Вход**: Все промежуточные Parquet-файлы + `models/*.json`

**Выход**: `reports/validation_report.md` + `reports/metrics.json`

**Метрики по фазам**:

| Фаза | Метрика | Метод | Цель |
|------|---------|------|------|
| 2 — Классификация | F1 (STEM vs non-STEM) | 200 размеченных вакансий | ≥ 0.92 |
| 3 — Кластеризация | Silhouette + DBI + bootstrap ARI | Внутренняя + экспертная | Silhouette > 0.4, ARI > 0.7 |
| 4 — Извлечение навыков | Precision@10 | Экспертная разметка топ-20 | ≥ 0.85 |
| 5 — Маппинг осей | Spearman ρ vs эксперт | 3 роли × 6 осей | ≥ 0.75 |
| 6 — Валидность модели | Структурная проверка | Автоматическая валидация JSON | 100% valid |
| Интеграция | Coverage | % вакансий с полной моделью | ≥ 85% |

## Структура проекта

```
krm/
├── config.yaml                     # Единый источник параметров
├── pyproject.toml
├── src/krm/
│   ├── __init__.py
│   ├── cli.py                      # Точка входа CLI
│   ├── config.py                   # Загрузка config.yaml
│   ├── lib/
│   │   ├── __init__.py
│   │   ├── io.py                   # read_parquet, write_parquet, duckdb helpers
│   │   ├── embeddings.py           # sentence-transformers wrapper с кэшированием
│   │   └── metrics.py              # F1, silhouette, ARI, Spearman ρ
│   ├── phase_1_collect.py          # HH.ru API scraper
│   ├── phase_2_classify.py         # STEM/IT/NON_STEM классификатор
│   ├── phase_3_roles.py            # Кластеризация названий должностей → РОЛИ
│   ├── phase_4_skills.py           # Извлечение навыков на роль
│   ├── phase_5_axes.py             # Zero-shot маппинг навыков → 6 осей
│   ├── phase_6_model.py            # Компетентностная модель + spider charts
│   └── phase_7_validate.py         # Полная валидация + отчёты
├── tests/
│   ├── conftest.py
│   ├── fixtures/                   # Малые размеченные выборки для CI
│   ├── test_phase_2.py
│   ├── test_phase_3.py
│   └── ...
├── data/                           # .gitignore'd, Parquet файлы
│   ├── raw/
│   ├── classified.parquet
│   ├── roles.parquet
│   ├── skills_per_role.parquet
│   └── axis_scores.parquet
├── models/                         # Компетентностные модели JSON
├── reports/                        # Валидация, spider charts PNG
└── docs/
```

## Технологический стек

| Компонент | Технология | Почему |
|-----------|-----------|--------|
| Эмбеддинги | `intfloat/multilingual-e5-large-instruct` | Лучший multilingual на MTEB, 1024-dim, instruction-following |
| Кластеризация | `umap-learn` + `hdbscan` | HDBSCAN: variable-density кластеры, шумовой кластер (-1); UMAP: global+local структура |
| NLI | `facebook/bart-large-mnli` | Zero-shot entailment, калиброванные вероятности |
| Skill taxonomy | ESCO v1.2.1 (CSV) | 13,939 канонических навыков, 28 языков, открытый стандарт ЕС |
| HH.ru API | `httpx` | Прямые HTTP-запросы к api.hh.ru |
| Хранение | DuckDB + Parquet (pyarrow) | DuckDB для SQL-запросов, Parquet для межфазного обмена |
| Визуализация | `matplotlib` | Публикационные PNG spider charts |
| Тестирование | `pytest` + `hypothesis` | Property-based тесты для edge cases кластеризации |
| Линтинг | `ruff` | E, F, I, N, UP, B, SIM, C4 |

## Конфигурация (`config.yaml`)

```yaml
pipeline:
  raw_dir: data/raw
  output_dir: data
  models_dir: models
  reports_dir: reports

collection:
  date_from: "2010-01-01"
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

axes:  # FIXED — never modified by pipeline
  - id: experimental
    label_ru: Экспериментальный опыт
    hypothesis: "This skill involves hands-on laboratory or field experimental work"
  - id: domain_knowledge
    label_ru: Предметные знания
    hypothesis: "This skill involves theoretical or conceptual understanding of a scientific domain"
  - id: management
    label_ru: Управление и коммуникации
    hypothesis: "This skill involves project management, team coordination, or scientific communication"
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

## Ключевые проектные решения

### 1. Фиксированные оси вместо data-driven axis discovery

**Было**: `AxisDiscoveryEngine` (926 строк) перебирает k=2..8, оценивает
bootstrap-валидированные метрики, выбирает оптимальное k.

**Стало**: Оси фиксированы в `config.yaml`. Навыки маппятся на оси через
zero-shot NLI. Это решение принято по требованию заказчика: оси должны
соответствовать 6 характеристикам исследователя, а не открываться из данных.

### 2. Кластеризация названий должностей вместо навыков

**Было**: `SkillClusterer` кластеризует навыки по co-occurrence → роли.

**Стало**: Эмбеддинги названий должностей → UMAP → HDBSCAN → роли.
Роль — это не кластер навыков, а кластер названий должностей.
Навыки извлекаются для роли постфактум (Фаза 4). Это соответствует тому,
как работодатели мыслят о ролях: они публикуют вакансию с названием должности,
а навыки — атрибут роли.

### 3. Междисциплинарные роли через мягкую кластеризацию

HDBSCAN `all_points_membership_vectors_` даёт вероятностное распределение
вакансии по кластерам. "Биофизик" может на 60% принадлежать физическому
кластеру и на 40% — биологическому. Это позволяет строить компетентностные
модели для междисциплинарных ролей без создания отдельных гибридных специальностей.

### 4. ESCO как единый словарь навыков

Вместо лемматизации pymorphy3 и точного совпадения — fuzzy matching с ESCO
(13,939 канонических названий навыков). Это решает проблему синонимии
("работа с микроскопом" → "microscopy", "микроскопия" → "microscopy").

### 5. Parquet как интерфейс между фазами

Каждая фаза читает Parquet с предыдущей, пишет Parquet для следующей.
Это даёт: естественное кэширование (doit по timestamp), отладку (инспекция
промежуточных файлов), воспроизводимость (зафиксированные артефакты).

### 6. 7-уровневая шкала: бакалавр → ГНС

**Было**: 5 уровней РАН (мнс → гнс).

**Стало**: 7 уровней (бакалавр, магистр, аспирант/мнс, нс, снс, внс, гнс).
Это позволяет строить компетентностные модели от undergraduate до senior
researcher и валидировать учебные программы бакалавриата/магистратуры.

## Рискованное предположение

**Самое рискованное**: что `multilingual-e5-large-instruct` адекватно
разделяет русскоязычные STEM-должности в embedding space (e.g., "биофизик"
vs "биохимик" vs "молекулярный биолог").

**Валидация до полной реализации**:
1. Собрать 150–200 русских STEM-вакансий из 5–6 заведомо разных ролей
2. Эмбеддинги → UMAP → HDBSCAN
3. Adjusted Rand Index против ground-truth меток
4. Критерий: ARI > 0.65. Если < 0.55 → переключиться на `deepvk/USER-bge-m3`
   или дообучить multilingual-e5 на русском STEM-корпусе
5. Затраты времени: ~4 часа (сбор 1ч, разметка 2ч, анализ 1ч)
