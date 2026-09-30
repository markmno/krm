"""Shared pytest fixtures for KRM pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import numpy as np


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def config_path() -> Path:
    return Path(__file__).parent.parent / "config.yaml"


@pytest.fixture
def sample_vacancies() -> pd.DataFrame:
    return pd.DataFrame({
        "vacancy_id": [f"v{i:03d}" for i in range(10)],
        "title": ["физик-экспериментатор", "химик-аналитик", "биоинформатик",
                  "младший научный сотрудник", "инженер-исследователь",
                  "лаборант химического анализа", "научный сотрудник (физика)",
                  "специалист по молекулярной биологии", "эколог-исследователь",
                  "python-разработчик"],
        "description": [
            "работа с лазерной установкой, оптический стол, интерферометрия",
            "хроматография, масс-спектрометрия, пробоподготовка",
            "анализ геномных данных, Python, R, секвенирование",
            "проведение экспериментов, обработка результатов",
            "разработка измерительного стенда, прототипирование",
            "титрование, pH-метрия, ведение лабораторного журнала",
            "измерение транспортных свойств, низкие температуры",
            "ПЦР, электрофорез, клонирование",
            "отбор проб воды, анализ на тяжелые металлы",
            "разработка backend на Django, PostgreSQL, REST API",
        ],
        "stem_category": ["STEM_RESEARCH"] * 6 + ["INTERDISCIPLINARY", "STEM_RESEARCH", "STEM_RESEARCH", "PURE_IT"],
    })


@pytest.fixture
def sample_roles() -> pd.DataFrame:
    return pd.DataFrame({
        "role_id": [0, 1, 2, 3],
        "role_label": ["физик-экспериментатор", "химик-аналитик",
                       "биоинформатик", "лаборант"],
        "top_titles": [
            json.dumps(["физик-экспериментатор", "инженер-исследователь"]),
            json.dumps(["химик-аналитик"]),
            json.dumps(["биоинформатик"]),
            json.dumps(["лаборант химического анализа"]),
        ],
        "member_count": [3, 2, 1, 2],
        "noise_flag": [False, False, False, False],
    })


@pytest.fixture
def sample_skills_per_role() -> pd.DataFrame:
    return pd.DataFrame({
        "role_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
        "skill_canonical_name": [
            "лазерная оптика", "интерферометрия", "написание научных статей",
            "хроматография", "масс-спектрометрия",
            "Python", "статистический анализ",
            "титрование", "pH-метрия",
        ],
        "esco_id": [f"S00{i}" for i in range(1, 10)],
        "frequency": [2, 1, 2, 2, 2, 1, 1, 2, 1],
        "tfidf_weight": [0.8, 0.4, 0.6, 0.9, 0.9, 0.7, 0.7, 0.8, 0.4],
    })


@pytest.fixture
def sample_characteristic_scores() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for role_id in range(4):
        for characteristic_id in [
            "domain_knowledge", "experimental", "data_analysis", "computational",
            "professional_texts", "t_profile", "management",
        ]:
            rows.append({
                "role_id": role_id,
                "characteristic_id": characteristic_id,
                "proficiency": round(float(rng.uniform(1.0, 5.0)), 2),
                "top_contributing_skills": json.dumps(["skill_a", "skill_b", "skill_c"]),
            })
    return pd.DataFrame(rows)
