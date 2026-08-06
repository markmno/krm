# Методология маппинга навыков на оси компетенций

## Проблема

Навык — это атомарная единица компетенции (например, "работа с электронным
микроскопом"). Ось компетенции — это агрегированная категория (например,
"Экспериментальный опыт"). Один навык может принадлежать нескольким осям.

Задача: отобразить каждый навык на одну или несколько осей с весами.

## Подход: Zero-shot Natural Language Inference (NLI)

### Модель

`facebook/bart-large-mnli` — fine-tuned BART на MultiNLI. Обучен определять
логические отношения между premise (навык) и hypothesis (описание оси):

- **entailment** (0) — навык логически следует из описания оси
- **neutral** (1) — навык не связан с осью
- **contradiction** (2) — навык противоречит оси

### Формулировка гипотез

Для каждой из 6 осей формулируется гипотеза на английском (рабочий язык NLI):

| Ось | Гипотеза |
|-----|---------|
| Экспериментальный опыт | This skill involves hands-on laboratory or field experimental work with scientific equipment |
| Предметные знания | This skill involves deep theoretical or conceptual understanding of a specific scientific domain |
| Управление и коммуникации | This skill involves project management, team leadership, grant writing, or scientific presentation |
| Научная литература | This skill involves searching, reading, writing, reviewing, or organizing scientific literature and technical documentation |
| Анализ данных | This skill involves statistical analysis, quantitative data processing, measurement error evaluation, or research methodology |
| Вычислительные методы | This skill involves computational modeling, numerical simulation, algorithm development, or high-performance computing |

### Процесс

1. Перевести русский навык на английский (multilingual-e5)
2. Подать в NLI: `premise = "{english skill} requires expertise in..."`, 6 гипотез
3. Извлечь entailment probability для каждой гипотезы
4. Нормализовать: `axis_weight = p_entail / (p_entail + p_neutral + p_contradiction)`

### Пример

Навык: "молекулярная динамика (GROMACS)"

NLI results:
- computational: entailment 0.87, neutral 0.10, contradict 0.03 → weight 0.87
- domain_knowledge: entailment 0.72 → weight 0.72
- experimental: entailment 0.12 → weight 0.12
- management: entailment 0.04 → weight 0.04
- literature: entailment 0.15 → weight 0.15
- data_analysis: entailment 0.22 → weight 0.22

Итог: навык вносит вклад в computational (0.87) и domain_knowledge (0.72).

## Агрегация на роль

Для каждой роли (кластера должностей):

```
proficiency[axis_j] = sum(skill_i.weight * skill_i.tfidf * nli_score[i][j])
                      / sum(skill_i.weight * skill_i.tfidf)
                      for all skills i in role

normalize: prof[axis_j] in [1, 5]
```

## Валидация

### Внутренняя

- Калибровка NLI: temperature scaling на 100 размеченных парах навык-ось
- Inter-rater: 3 эксперта размечают 20 навыков, Cohen's kappa >= 0.7

### Внешняя

- Spearman rank correlation экспертных оценок и модельных: rho >= 0.75
- 3 роли x 6 осей = 18 суждений

### Edge cases

| Случай | Обработка |
|--------|----------|
| p_entail < 0.1 для всех осей | Навык в "uncatalogued", ручная классификация |
| p_entail > 0.7 для 3+ осей | Навык general-purpose (e.g., "работа в команде"), равномерно распределяется |
| Навык не найден в ESCO | Fuzzy matching с cosine > 0.7, иначе uncatalogued |

## Почему NLI, а не classification?

1. **Zero-shot**: не нужно размеченных данных для обучения классификатора
2. **Calibrated probabilities**: BART-MNLI даёт хорошо калиброванные вероятности
3. **Interpretable**: каждая гипотеза — человекочитаемое утверждение
4. **Extensible**: добавить новую ось = добавить гипотезу, без переобучения

## Почему не regex (как было в v0.1)?

Regex-правила (категории "Programming", "Experiment", "Computational"):

- Требуют ручного составления и поддержки списков триггерных слов
- Не улавливают синонимы ("микроскоп" vs "микроскопия" vs "electron microscopy")
- Не дают весов — навык либо в категории, либо нет (hard assignment)
- Не масштабируются на новые домены
- NLI решает все эти проблемы через семантическое понимание
