# КРМ — Компетентностно-Ролевая Модель STEM-рынка труда

Инструмент для построения компетентностно-ролевых моделей на основе данных
российского рынка труда (HH.ru). Отличается от предыдущей версии `hh-competency`
полностью data-driven подходом: роли обнаруживаются из названий вакансий (а не
хардкодятся), навыки извлекаются через NLP-эмбеддинги (а не regex-правила), оси
компетенций фиксированы на основе характеристик исследователя (а не открываются
статистически).

> **Статус**: перепроектирование. Заменяет `hh-competency` v0.1.

## Отличия от hh-competency v0.1

| Аспект | hh-competency v0.1 | КРМ (новая версия) |
|--------|-------------------|---------------------|
| Оси компетенций | Data-driven: AgglomerativeClustering + bootstrap validation (k=2..8) | **Фиксированы**: 7 характеристик исследователя |
| Классификация навыков по осям | 234 regex-правила (9 категорий) | Zero-shot NLI: multilingual-e5-large + entailment |
| Обнаружение ролей | Кластеризация навыков по co-occurrence | **Кластеризация названий должностей**: embeddings + UMAP + HDBSCAN |
| Уровневая шкала | 5 уровней РАН (мнс→гнс) | **Бакалавр → ГНС** (7 уровней: бакалавр, магистр, аспирант, мнс, нс, снс, внс, гнс) |
| Извлечение навыков | pymorphy3 + лемматизация + точное совпадение | sentence-transformers + ESCO taxonomy (13,939 навыков) + fuzzy matching |
| Валидация учебной программы | 25 хардкоженных курсов + regex | **Реальные силлабусы ИТМО** + семантический matching |
| STEM-vs-IT фильтр | Отсутствует | HH.ru API pre-filter + zero-shot NLI refinement |
| Хранение | DuckDB (JSON blob) | DuckDB + Parquet (эмбеддинги) |
| Оркестрация | 8-фазный workflow (Typer) | 7-фазный пайплайн |

## 7 фиксированных осей компетенций

| # | id | Ось | EN | Гипотеза NLI |
|---|----|-----|----|-------------|
| 1 | `domain_knowledge` | Доменная база | Domain Knowledge | "This skill requires deep theoretical or applied knowledge in a specific scientific or technical domain." |
| 2 | `experimental` | Эксперимент | Experimental Work | "This skill involves hands-on laboratory or field experimental work with equipment, organisms, or materials." |
| 3 | `data_analysis` | Анализ данных | Data Analysis | "This skill involves analyzing data, performing statistical tests, or applying quantitative methods." |
| 4 | `computational` | Вычислительные методы | Computational Methods | "This skill involves computational modeling, programming, algorithm development, or numerical methods." |
| 5 | `professional_texts` | Профессиональные тексты | Professional Texts | "This skill involves writing, reading, or working with professional and scientific texts, documentation, publications, and reports." |
| 6 | `t_profile` | T-профиль | T-Profile | "This skill reflects broad erudition and interdisciplinary breadth — knowledge that spans multiple STEM fields beyond a single specialty." |
| 7 | `management` | Управление | Management & Organization | "This skill involves managing teams, projects, budgets, or organizational processes." |

## Структура документации

| Документ | Описание |
|----------|----------|
| [architecture.md](architecture.md) | Архитектура: 7-фазный пайплайн, стек, поток данных, структура кода |
| [methodology/scientific-workflow.md](methodology/scientific-workflow.md) | Научная методология: 7 фаз, статистическая валидация, метрики качества |
| [research/competency-model-frameworks.md](research/competency-model-frameworks.md) | Обзор фреймворков: SFIA 9, ESCO, РАН, FGOS, RSE, ISCB, Big Tech |
| [research/russian-job-market-data.md](research/russian-job-market-data.md) | Источники данных: HH.ru API, NLP-инструменты, датасеты |
| [user-guide.md](user-guide.md) | Руководство пользователя (будет обновлено) |

## Быстрый старт

```bash
# Установка
uv sync

# Полный пайплайн: сбор → классификация → роли → навыки → оси → модель → валидация
krm all

# Поэтапный запуск
krm collect                              # Сбор вакансий с HH.ru
krm classify                              # Классификация STEM/IT
krm roles                                 # Кластеризация ролей
krm skills                                # Извлечение навыков (ESCO)
krm axes                                  # Маппинг навыков на 7 осей
krm model                                 # Построение моделей + spider charts
krm validate                              # Валидация + отчёты
```

## Технологический стек

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| Язык | Python ≥ 3.11 | Основной язык |
| Пакетный менеджер | uv | Зависимости и окружение |
| CLI | sys.argv парсер | Интерфейс командной строки |
| HTTP | httpx + asyncio | Сбор вакансий с HH.ru |
| Хранение | DuckDB + Parquet (pyarrow) | База данных + файловый обмен |
| Эмбеддинги | intfloat/multilingual-e5-large-instruct | 1024-мерные эмбеддинги для кластеризации и классификации |
| Кластеризация | umap-learn + hdbscan | Обнаружение ролей из названий должностей |
| NLI | facebook/bart-large-mnli | Zero-shot классификация навыков по осям |
| Skill taxonomy | ESCO v1.2.1 (13,939 навыков) | Нормализация названий навыков |
| Визуализация | matplotlib | Spider charts, bar charts |
| Статистика | scipy + scikit-learn | Silhouette, DBI, bootstrap ARI, Spearman ρ |
| Тестирование | pytest + hypothesis | Модульное и property-based тестирование |
| Линтинг | ruff | Качество кода |
