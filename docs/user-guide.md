# Руководство пользователя KRM

**KRM** (Компетентностно-Ролевая Модель) — инструмент командной строки для
построения компетентностно-ролевых моделей на основе данных о вакансиях с HH.ru.
Предназначен для анализа рынка труда, сравнения требований работодателей с
учебными программами и построения spider-диаграмм компетенций.

---

## Установка

Требуется Python 3.11+. Рекомендуемый менеджер окружения — [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd krm
uv sync
```

После установки проверьте, что CLI доступен:

```bash
krm --version
```

Ожидаемый вывод: `krm v0.2.0`

---

## Настройка

### Файл config.yaml

Конфигурационный файл `config.yaml` находится в корне проекта. В нём определены:

- **Ключевые слова сбора** — поисковые запросы для HH.ru (физик, химик, биолог и др.);
- **Параметры классификации** — пороги STEM/IT классификатора;
- **Параметры кластеризации** — модель эмбеддингов, HDBSCAN параметры;
- **Параметры извлечения навыков** — версия ESCO, пороги fuzzy matching;
- **7 осей компетенций** — названия и NLI-гипотезы;
- **4 soft-компетенции** — названия, NLI-гипотезы и per-archetype baseline;
- **Параметры тестирования** — размеры выборок, bootstrap итерации.

Подробнее см. `docs/methodology/config-guide.md`.

### HH.ru API

Для сбора вакансий используется OAuth2-токен HH.ru:

1. Зарегистрируйте приложение на [dev.hh.ru](https://dev.hh.ru/).
2. Получите `client_id` и `client_secret`.
3. Установите переменные окружения:

```bash
export HH_CLIENT_ID=ваш_client_id
export HH_CLIENT_SECRET=ваш_client_secret
```

Без токена доступен только анализ ранее собранных данных (флаг `--skip-collect`).

### Хранение данных

Все промежуточные данные хранятся в формате Parquet в папке `data/`:

- `data/raw/` — сырые ответы HH.ru
- `data/classified.parquet` — результат классификации STEM/IT
- `data/roles.parquet` — обнаруженные роли (кластеры должностей)
- `data/characteristics.parquet` — по-вакансийные оценки характеристик (Фаза 2.5)
- `data/skills_per_role.parquet` — навыки по ролям
- `data/vacancy_roles.parquet` — маппинг `vacancy_id → role_id`
- `data/characteristic_scores.parquet` — оценки по 7 осям компетенций
- `data/skill_characteristic_scores.parquet` — матрица (навык × ось) NLI-оценок
- `data/role_archetypes.parquet` — архетип каждой роли
- `data/soft_scores.parquet` — оценки по 4 soft-компетенциям

Результаты сохраняются в `models/` (JSON-модели) и `reports/` (4 диаграммы на роль, отчёты).

---

## Команды

KRM предоставляет отдельные команды для каждой фазы 7-фазного пайплайна:

| Команда    | Фаза | Назначение                                      |
|------------|------|-------------------------------------------------|
| `collect`  | 1    | Сбор исторических STEM-вакансий с HH.ru        |
| `classify` | 2    | Классификация вакансий на STEM/IT/non-STEM     |
| `phase25`  | 2.5  | По-вакансийный scoring характеристик (7 осей)  |
| `roles`    | 3    | Кластеризация названий должностей → роли       |
| `skills`   | 4    | Извлечение навыков на роль (ESCO словарь)      |
| `axes`     | 5    | Zero-shot NLI маппинг навыков на 7 осей        |
| `soft`     | 5b   | Классификация архетипов + soft-компетенции     |
| `model`    | 6    | Построение компетентностных моделей + 4 диаграммы |
| `validate` | 7    | Валидация и отчёты по метрикам                 |
| `all`      | 1–7  | Полный пайплайн от сбора до отчёта             |

---

### `krm collect` — сбор вакансий

```bash
krm collect
```

Собирает вакансии с HH.ru API по ключевым словам из `config.yaml`.
Поддерживает инкрементальный сбор (пропускает уже загруженные вакансии).

Использует следующие параметры из `config.yaml`:
- `collection.keywords` — список поисковых запросов (физик, химик, биолог и др.)
- `collection.professional_roles: [79]` — R&D вакансии
- `collection.categories: [25]` — Наука/Образование
- `collection.exclude_roles: [96]` — исключение ИТ/Телеком
- `collection.date_from: "2010-01-01"` — начальная дата
- `collection.rate_limit_rps: 2` — ограничение запросов/сек

Результат: Parquet-файлы в `data/raw/`.

---

### `krm classify` — классификация STEM/IT

```bash
krm classify
```

4-уровневый классификатор:

| Tier | Метод | Что делает |
|------|------|-----------|
| 1 | API rule | IT роли (96, 123–126) → PURE_IT |
| 2 | Keyword triage | Programm/Razrabotchik без STEM-слов → PURE_IT |
| 3 | Zero-shot NLI (bart-large-mnli) | Три гипотезы: STEM research, IT, Admin |
| 4 | Interdisciplinary catch | STEM + IT > порог → INTERDISCIPLINARY |

Результат: `data/classified.parquet` с колонкой `stem_category` (STEM_RESEARCH, PURE_IT, NON_STEM, INTERDISCIPLINARY).

---

### `krm roles` — обнаружение ролей

```bash
krm roles
```

1. Эмбеддинги всех уникальных названий должностей через `multilingual-e5-large-instruct`
2. UMAP снижение размерности (16 компонент)
3. HDBSCAN кластеризация (variable-density кластеры + шумовой кластер -1)
4. Мягкое назначение для междисциплинарных ролей

Результат: `data/roles.parquet` (role_id, role_label, top_titles, member_count, centroid_embedding).

Валидация: Silhouette score > 0.4, bootstrap ARI стабильность.

---

### `krm skills` — извлечение навыков

```bash
krm skills
```

- Dictionary-based matching с ESCO v1.2.1 (13,939 канонических навыков)
- Russian Profstandart skills (40.008, 40.011)
- Fuzzy matching: cosine similarity > 0.7
- TF-IDF взвешивание на роль
- Некаталогизированные навыки сохраняются отдельно

Результат: `data/skills_per_role.parquet` (role_id, skill_canonical_name, esco_id, frequency, tfidf_weight).

Метрика: Precision@10 ≥ 0.85.

---

### `krm axes` — маппинг на 7 осей

```bash
krm axes
```

Zero-shot NLI маппинг каждого навыка на 7 фиксированных осей:

| Ось | Описание |
|-----|----------|
| Доменная база | Глубокие теоретические/прикладные знания в научной области |
| Эксперимент | Лабораторная/полевая экспериментальная работа |
| Анализ данных | Статистика, обработка данных, количественные методы |
| Вычислительные методы | Моделирование, программирование, алгоритмы |
| Профессиональные тексты | Работа с научными текстами, документацией, публикациями |
| T-профиль | Междисциплинарная широта и эрудированность |
| Управление | Управление командами, проектами, бюджетами, процессами |

Алгоритм: мягкое распределение (один навык — несколько осей), агрегация на роль,
нормализация 1–5.

Результат: `data/characteristic_scores.parquet` (role_id, axis_name, proficiency_1_to_5, top_contributing_skills) и `data/skill_characteristic_scores.parquet` (полная матрица навык × ось).

Валидация: Spearman ρ ≥ 0.75 (экспертные оценки).

---

### `krm soft` — архетипы и soft-компетенции

```bash
krm soft
```

Классифицирует каждую роль по 6 архетипам (Техник, Исследователь, Методолог,
Инженер, Ведущий специалист, Гибрид) и оценивает 4 soft-компетенции:

| Soft-компетенция | Описание |
|------------------|----------|
| Мышление | Критическое мышление, аналитика, решение проблем |
| Командное Взаимодействие | Командная работа, координация, сотрудничество |
| Ответственность и Лидерство | Ответственность, лидерство, принятие решений, менторство |
| Профессиональная культура | Этика, научная добросовестность, стандарты |

Оценка: NLI каждой вакансии против 4 гипотез → средний entailment. Оси, не
набравшие порог confidence/числа описаний, берут значение из **per-archetype
baseline** (задан в `config.yaml`, шкала 1–5).

Результат: `data/role_archetypes.parquet` (role_id, archetype, super_fractions) и `data/soft_scores.parquet` (role_id, soft_id, proficiency).

---

### `krm model` — компетентностные модели

```bash
krm model
```

Для каждой обнаруженной роли строит:

- **JSON-модель**: role_label, архетип, топ-10 должностей, per-axis proficiency (1–5), топ-3 навыка-драйвера на ось, 4 soft-компетенции, гистограмма опыта, топ-15 навыков
- **4 диаграммы на роль** (matplotlib PNG):
  1. Experience — гистограмма лет опыта (`reports/<domain>/experience/{role_id}.png`)
  2. Hard competences — 7-осевая spider chart (`reports/<domain>/hard/{role_id}.png`)
  3. Soft competences — 4-осевая spider chart (`reports/<domain>/soft/{role_id}.png`)
  4. Skills — топ-15 баров, сегментированных по осям (`reports/<domain>/skills/{role_id}.png`)

Результат: `models/{role_id}.json` и 4 PNG на роль в `reports/<domain>/`.

---

### `krm validate` — валидация и отчёты

```bash
krm validate
```

Рассчитывает метрики по всем фазам:

| Фаза | Метрика | Цель |
|------|---------|------|
| 2 — Классификация | F1 (STEM vs non-STEM) | ≥ 0.92 |
| 3 — Кластеризация | Silhouette score | > 0.4 |
| 4 — Извлечение навыков | Precision@10 | ≥ 0.85 |
| 5 — Маппинг осей | Spearman ρ | ≥ 0.75 |
| Интеграция | Coverage (% вакансий с полной моделью) | ≥ 85% |

Результат: `reports/validation_report.md` и `reports/metrics.json`.

---

### `krm all` — полный пайплайн

```bash
krm all
```

Запускает все фазы последовательно: **collect → classify → phase25 → roles → skills → axes → soft → model → validate**.

Для пропуска сбора (использовать существующие данные):

```bash
krm all --skip-collect
```

---

## Типичные сценарии

### Сценарий 1: Первый запуск с нуля

1. Настройте `config.yaml`: ключевые слова сбора, пороги классификации.
2. Получите OAuth2-токен HH.ru и установите переменные окружения.
3. Запустите полный пайплайн:

```bash
krm all
```

4. Изучите 4 диаграммы на роль в `reports/<domain>/` и валидационный отчёт в `reports/validation_report.md`.

### Сценарий 2: Анализ существующих данных

Если данные HH.ru уже собраны, минуйте сбор и перестройте модель:

```bash
krm classify && krm phase25 && krm roles && krm skills && krm axes && krm soft && krm model && krm validate
```

Или одной командой:

```bash
krm all --skip-collect
```

### Сценарий 3: Исследовательский анализ (одна фаза)

Кластеризация с новыми параметрами:

1. Измените `config.yaml`: `roles.hdbscan_min_cluster_size` с 15 на 10.
2. Перезапустите только фазы 3→7:

```bash
krm roles && krm skills && krm axes && krm soft && krm model && krm validate
```

### Сценарий 4: Добавление специальности

1. Добавьте ключевые слова в `config.yaml` `collection.keywords`:

```yaml
collection:
  keywords: [физик, химик, биолог, эколог, геолог]  # добавлен геолог
```

2. Докачайте новые данные:

```bash
krm collect
```

3. Перестройте модель:

```bash
krm all --skip-collect
```

---

## Выходные форматы

| Формат | Описание | Где |
|--------|----------|-----|
| Parquet | Промежуточные данные между фазами | `data/*.parquet` |
| JSON | Компетентностные модели ролей | `models/{role_id}.json` |
| PNG | 4 диаграммы на роль (публикационное качество) | `reports/<domain>/hard\|soft\|experience\|skills/{role_id}.png` |
| Markdown | Валидационный отчёт с метриками | `reports/validation_report.md` |

---

## Ограничения

- **HH.ru API**: максимум 2 000 вакансий на поисковый запрос (обход — множественные запросы, дедупликация через DuckDB).
- **Rate limit**: ~2 запроса в секунду, экспоненциальный backoff при 429.
- **OAuth2**: обязателен для сбора данных; без токена — `krm all --skip-collect`.
- **Модели NLP**: при первом запуске загружаются `multilingual-e5-large-instruct` (~2.2 ГБ) и `bart-large-mnli` (~1.6 ГБ).
- **Регион**: по умолчанию поиск по России (area=113).

---

## Дополнительная информация

- Архитектура пайплайна: `docs/architecture.md`
- Методология: `docs/methodology/scientific-workflow.md`
- Стратегия тестирования: `docs/methodology/testing-strategy.md`
- Маппинг навыков на оси: `docs/methodology/skill-mapping.md`
- Руководство по конфигурации: `docs/methodology/config-guide.md`
- Конфигурация: `config.yaml` (корень проекта)
- Данные: `data/` (Parquet, .gitignored)
