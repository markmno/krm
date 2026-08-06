# Методология: Научный workflow

> Дата: 2026-08-05

## 1. Постановка задачи

**Проблема.** Российский STEM-рынок труда использует нестандартизированные
названия должностей. Единое название "научный сотрудник" может означать
экспериментатора, теоретика или менеджера проектов. Отсутствует эмпирическое
отображение между названиями должностей, навыками и компетенциями.

**Научный вопрос.** Можно ли обнаружить латентную ролевую структуру из названий
STEM-должностей через embedding-based кластеризацию?

**Цель.** Построить пайплайн: сбор HH.ru вакансий (не IT) → классификация 
STEM/IT → кластеризация должностей в роли → извлечение навыков → маппинг на 6
осей → компетентностные модели.

## 2. 7 фаз

### Фаза 1: Сбор вакансий

- HH.ru API через `kirillzhosul/hhru` SDK
- Поиск по STEM-ключевым словам, `professional_role=79` (R&D), категория 25
- Исключение `professional_role=96` (ИТ/Телеком)
- Инкрементальный сбор, дедупликация через DuckDB
- Ограничение API: 2000/запрос → множественные запросы

### Фаза 2: Классификация STEM/IT/non-STEM

4-уровневый классификатор:

| Tier | Метод | Назначение |
|------|------|-----------|
| 1 | API rule: prole in {96, 123-126} | IT отсекается сразу |
| 2 | Keyword: [Pp]rogramm/Razrabotchik без STEM-слов | Ложноположительные IT |
| 3 | Zero-shot NLI: bart-large-mnli, 3 гипотезы | Ambiguous (~15%) |
| 4 | Interdisciplinary: STEM+IT > 0.6 | Сохраняется |

Категории: STEM_RESEARCH | PURE_IT | NON_STEM | INTERDISCIPLINARY

Метрика: F1 >= 0.92 на размеченной выборке (200 вакансий, Cohen's kappa).

Гипотезы для NLI:
- "This job is a scientific research position requiring experimental"
- "This job is a software engineering or IT development role"
- "This job is an administrative or non-research academic role"

### Фаза 3: Кластеризация должностей в роли

**Алгоритм:**
1. Эмбеддинги: `intfloat/multilingual-e5-large-instruct` (1024-dim)
2. UMAP: n_components=16, metric=cosine, min_dist=0.0
3. HDBSCAN: min_cluster_size=15, cluster_selection_epsilon=0.25
4. Мягкая кластеризация: `all_points_membership_vectors_`
5. Именование: топ-3 центральных названия

**Почему HDBSCAN, а не K-means:**
- Variable-density кластеры (одни роли разнообразнее других)
- Шумовой кластер -1 (не форсируем выбросы)
- Не требует задания k заранее
- Мягкое назначение для междисциплинарных вакансий

**Метрики:** Silhouette (>0.4), Davies-Bouldin index, Bootstrap ARI (100 итераций)

### Фаза 4: Извлечение навыков на роль

- Dictionary-based matching с ESCO v1.2.1 (13,939 навыков)
- Russian Profstandart skills (40.008, 40.011)
- Fuzzy matching: cosine similarity > 0.7
- TF-IDF взвешивание на роль
- Некаталогизированные навыки сохраняются отдельно

Метрика: Precision@10 >= 0.85 (экспертная разметка)

### Фаза 5: Маппинг на 6 осей

6 фиксированных осей (из config.yaml, НЕ обнаруживаются):

| Ось | NLI-гипотеза |
|-----|-------------|
| Экспериментальный опыт | hands-on laboratory or field experimental work |
| Предметные знания | theoretical understanding of a scientific domain |
| Управление и коммуникации | project management, coordination, communication |
| Научная литература | reading, writing, organizing scientific documents |
| Анализ данных | statistical analysis, data processing, quantitative |
| Вычислительные методы | computational modeling, simulation, algorithm |

Алгоритм: zero-shot NLI на каждый навык → мягкое распределение → 
агрегация на роль → нормализация шкала 1-5.

Валидация: Spearman rho >= 0.75 vs экспертные оценки.

### Фаза 6: Компетентностные модели

- Per-role JSON: топ-10 должностей, 7-уровневая шкала, per-axis scores
- Spider charts: plotly (интерактивный HTML) + matplotlib (PNG)
- Кросс-референсы: Profstandart 40.008, 40.011

### Фаза 7: Валидация

| Фаза | Метрика | Цель |
|------|---------|------|
| 2 | F1 (STEM vs non) | >= 0.92 |
| 3 | Silhouette | > 0.4 |
| 4 | Precision@10 | >= 0.85 |
| 5 | Spearman rho | >= 0.75 |
| Интеграция | Coverage | >= 85% |

## 3. Статистический workflow

### Этап 0: Разведочный анализ

- Распределение названий должностей (топ-50)
- Ключевые слова STEM-вакансий (n-grams 2-5)
- Распределение длин описаний
- Сезонность и тренды

### Этап 1: Предобработка

- STEM/IT классификация (Фаза 2)
- Очистка: удаление HTML, нормализация whitespace
- Лемматизация: pymorphy3 (только для keyword matching, не для embeddings)

### Этап 2: Построение ролей

- Embedding всех уникальных названий должностей
- UMAP: выбор n_components через trustworthiness metric
- HDBSCAN: grid search min_cluster_size
- Bootstrap стабильность: 100 итераций на 80% подвыборках
- Экспертная валидация: adjusted Rand index (150-200 вакансий)

### Этап 3: Обогащение ролей навыками

- ESCO матчинг с порогом
- TF-IDF взвешивание
- Precision@10 через экспертную разметку

### Этап 4: Компетентностные профили

- NLI на каждый навык (6 гипотез)
- Агрегация взвешенных энтейлментов
- Экспертная валидация (Spearman rho)

## 4. Ключевые проектные решения

1. **Роли = кластеры должностей, не навыков.** Навыки — атрибут роли,
   извлекаемый постфактум.

2. **Оси фиксированы, навыки обнаружены.** Никаких regex-правил. Zero-shot
   NLI для маппинга.

3. **Мягкая кластеризация для междисциплинарности.** HDBSCAN membership
   vectors позволяют вакансии принадлежать нескольким кластерам.

4. **ESCO как единый словарь.** 13,939 канонических названий, 28 языков,
   открытый стандарт.

5. **7-уровневая шкала: бакалавр → ГНС.** От undergraduate до senior
   researcher.

6. **Parquet как DAG-интерфейс.** Файловая система — контракт между фазами.

## 5. Риски и валидация

**Самый рискованный этап:** кластеризация. Проверяется ДО полной реализации:
- 150-200 размеченных вакансий из 5-6 заведомо разных ролей
- Embeddings -> UMAP -> HDBSCAN -> ARI против ground truth
- Если ARI < 0.55: переключиться на `deepvk/USER-bge-m3`

**Другие риски:**
- HH.ru API rate limiting: экспоненциальный backoff + кэширование
- Размер выборки: мониторинг coverage, таргет >500 на роль
- ESCO покрытие русских терминов: fuzzy matching + ручная классификация
- NLI calibration: temperature scaling на размеченных парах
