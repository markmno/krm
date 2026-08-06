# Стратегия тестирования

## Философия

Каждая фаза пайплайна тестируется на трёх уровнях:

| Уровень | Объём | Время | Покрытие |
|---------|-------|-------|----------|
| **Unit** | Функции/классы изолированно | < 5s | Все публичные API |
| **Integration** | 2+ фаз с реальными файлами | < 60s | I/O и контракты |
| **Smoke** | Полный пайплайн на малых данных | < 5min | End-to-end |

## Unit-тесты

### Фаза 2: Классификация

```python
# tests/test_phase_2.py

def test_classify_pure_it_via_api_rule():
    """Вакансия с professional_role=96 -> PURE_IT без вызова NLI."""
    ...

def test_classify_stem_via_nli():
    """Вакансия физика-экспериментатора -> STEM_RESEARCH."""
    ...

def test_classify_interdisciplinary():
    """Биоинформатик со STEM+IT гипотезами > 0.6 -> INTERDISCIPLINARY."""
    ...
```

### Фаза 3: Кластеризация (самая важная)

```python
def test_embedding_dimension():
    """multilingual-e5-large даёт 1024-dim векторы."""
    ...

def test_hdbscan_identifies_known_clusters():
    """Заведомо разные должности попадают в разные кластеры."""
    ...

def test_noise_cluster_negative_one():
    """Выбросы помечаются как -1, не форсируются в кластеры."""
    ...

def test_soft_membership_is_probability():
    """membership vectors суммируются в ~1."""
    ...
```

### Фаза 5: Маппинг осей

```python
def test_nli_experimental_skill():
    """Навык 'работа с лазерной установкой' -> high experimental, low computational."""
    ...

def test_nli_computational_skill():
    """Навык 'численное моделирование Монте-Карло' -> high computational."""
    ...

def test_proficiency_normalization():
    """Scores всегда в диапазоне 1-5."""
    ...
```

## Интеграционные тесты

### Контракты Parquet

```python
def test_phase_2_output_schema_matches_phase_3_input():
    """classified.parquet содержит все колонки, ожидаемые фазой 3."""
    ...

def test_role_ids_are_stable():
    """Повторный запуск фазы 3 с теми же данными даёт те же role_id."""
    ...
```

### Full pipeline smoke

```python
@pytest.mark.smoke
def test_full_pipeline_on_fixtures():
    """Запуск всех 7 фаз на fixtures/ с 50 вакансиями."""
    # Collect -> Classify -> Roles -> Skills -> Axes -> Model -> Validate
    ...
    assert report["coverage"] >= 0.8
```

## Тестовые фикстуры

`tests/fixtures/` содержит:

```
fixtures/
├── raw_vacancies.parquet          # 50 реальных вакансий (анонимизированных)
├── expected_classification.csv    # Ручная разметка STEM/IT/non-STEM
├── expected_roles.csv             # Ручная разметка: должность -> роль
├── expected_axis_scores.csv       # Экспертные оценки: роль -> 6 осей
└── esco_skills_sample.csv         # Подмножество ESCO (500 навыков)
```

Фикстуры неизменны, коммитятся в репозиторий.

## Свойства Hypothesis

```python
from hypothesis import given, strategies as st

@given(title=st.text(min_size=3, max_size=100))
def test_embedding_is_length_invariant(title):
    """Любое название должности даёт вектор фиксированной размерности."""
    vec = get_embedding(title)
    assert len(vec) == 1024

@given(scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=6, max_size=6))
def test_proficiency_is_1_to_5(scores):
    """Любой набор NLI-scores нормализуется в 1-5."""
    prof = normalize_to_proficiency(scores)
    assert 1.0 <= min(prof) <= max(prof) <= 5.0
```

## Скорость в CI

| Пакет тестов | Количество | Время | Запуск |
|-------------|-----------|-------|--------|
| Unit | ~40 | 10s | Каждый коммит |
| Integration | ~8 | 45s | Каждый PR |
| Smoke | 1 | 3min | Перед merge |
| Hypothesis | ~6 | 30s | Nightly |

## Договорённости

1. Никакого monkey-patching HH.ru API — тесты используют фикстуры
2. Модели NLP не загружаются в CI — тесты используют pre-computed эмбеддинги
3. Валидационные метрики (F1, ARI, rho) считаются на реальных размеченных данных
4. Hypothesis property-based тесты ловят edge cases (пустые строки, unicode, длины)
