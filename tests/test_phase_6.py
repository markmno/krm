"""Tests for phase_6_model.py — competency models and weight mapping."""

import json
from pathlib import Path

import pandas as pd
import pytest

from krm.config import Config
from krm.phase_6_model import (
    _COMPETENCY_LEVELS,
    _build_single_model,
    _parse_json_field,
    _weight_to_threshold,
    build_models,
)

# ---------------------------------------------------------------------------
# 1. Constants — _COMPETENCY_LEVELS
# ---------------------------------------------------------------------------


class TestCompetencyLevels:
    """Verify the 7-level scale structure, order, and weight progression."""

    def test_has_exactly_seven_levels(self):
        assert len(_COMPETENCY_LEVELS) == 7

    def test_level_ids_are_sequential(self):
        for i, level in enumerate(_COMPETENCY_LEVELS):
            assert level["id"] == i

    def test_labels_match_expected_order(self):
        expected = [
            "Бакалавр",
            "Магистр",
            "Аспирант / МНС",
            "Научный сотрудник",
            "Старший научный сотрудник",
            "Ведущий научный сотрудник",
            "Главный научный сотрудник",
        ]
        actual = [lvl["label"] for lvl in _COMPETENCY_LEVELS]
        assert actual == expected

    def test_weights_are_strictly_increasing(self):
        weights = [lvl["weight"] for lvl in _COMPETENCY_LEVELS]
        for i in range(1, len(weights)):
            assert weights[i] > weights[i - 1]

    def test_weights_match_expected_values(self):
        expected_weights = [0.10, 0.25, 0.40, 0.60, 0.80, 0.95, 1.00]
        actual = [lvl["weight"] for lvl in _COMPETENCY_LEVELS]
        assert actual == expected_weights

    def test_first_weight_is_zero_point_one_zero(self):
        assert _COMPETENCY_LEVELS[0]["weight"] == 0.10

    def test_last_weight_is_one_point_zero_zero(self):
        assert _COMPETENCY_LEVELS[6]["weight"] == 1.00

    def test_all_levels_have_required_keys(self):
        required = {"id", "label", "weight"}
        for level in _COMPETENCY_LEVELS:
            assert set(level.keys()) == set(level.keys()) | required
            assert set(level.keys()) >= required


# ---------------------------------------------------------------------------
# 2. _weight_to_threshold — map 0–1 weight onto 1–5 proficiency
# ---------------------------------------------------------------------------


class TestWeightToThreshold:
    """Boundary and midpoint tests for the linear weight→threshold mapping."""

    # -- exact values from the task specification --
    @pytest.mark.parametrize(
        ("weight", "expected"),
        [
            (0.00, 1.0),
            (0.10, 1.4),
            (0.25, 2.0),
            (0.40, 2.6),
            (0.50, 3.0),
            (0.60, 3.4),
            (0.80, 4.2),
            (0.95, 4.8),
            (1.00, 5.0),
        ],
    )
    def test_exact_values(self, weight, expected):
        result = _weight_to_threshold(weight)
        assert result == pytest.approx(expected)

    def test_monotonic(self):
        """Threshold must increase strictly with weight."""
        results = [
            _weight_to_threshold(w)
            for w in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
        ]
        for i in range(1, len(results)):
            assert results[i] > results[i - 1]

    def test_range_lower_bound(self):
        assert _weight_to_threshold(0.0) == 1.0

    def test_range_upper_bound(self):
        assert _weight_to_threshold(1.0) == 5.0

    def test_midpoint_is_three(self):
        assert _weight_to_threshold(0.5) == pytest.approx(3.0)

    def test_negative_weight_clamped(self):
        """Negative weight produces value below _PROFICIENCY_MIN (not clamped)."""
        result = _weight_to_threshold(-0.2)
        assert result < 1.0

    def test_above_one_weight_exceeds_max(self):
        """Weight > 1 produces value above _PROFICIENCY_MAX (not clamped)."""
        result = _weight_to_threshold(1.5)
        assert result > 5.0


# ---------------------------------------------------------------------------
# 3. _parse_json_field — safe JSON-string/list parsing
# ---------------------------------------------------------------------------


class TestParseJsonField:
    """Parsing of JSON-string columns and list pass-through."""

    def test_valid_json_string_returns_list(self):
        result = _parse_json_field('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_empty_json_array(self):
        result = _parse_json_field("[]")
        assert result == []

    def test_list_passthrough(self):
        """Already a list — returned as-is."""
        result = _parse_json_field(["x", "y"])
        assert result == ["x", "y"]

    def test_empty_list_passthrough(self):
        result = _parse_json_field([])
        assert result == []

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_field("{invalid")

    def test_none_returns_empty_list(self):
        result = _parse_json_field(None)
        assert result == []

    def test_non_string_non_list_returns_empty(self):
        result = _parse_json_field(42)
        assert result == []

    def test_string_with_spaces(self):
        result = _parse_json_field('  ["a", "b"]  ')
        assert result == ["a", "b"]

    def test_string_with_unicode(self):
        result = _parse_json_field('["физик", "химик"]')
        assert result == ["физик", "химик"]


# ---------------------------------------------------------------------------
# 4. _build_single_model — competency model construction
# ---------------------------------------------------------------------------

ROLE_ROW = pd.Series(
    {
        "role_id": 5,
        "role_label": "физик-экспериментатор",
        "top_titles": json.dumps(
            ["физик-экспериментатор", "инженер-исследователь"]
        ),
        "member_count": 42,
    }
)

CHARACTERISTIC_ROWS_DF = pd.DataFrame(
    {
        "characteristic_id": ["experimental", "data_analysis", "computational"],
        "proficiency": [4.2, 1.8, 3.0],
        "top_contributing_skills": [
            json.dumps(["лазерная оптика", "интерферометрия", "вакуумная техника", "extra"]),
            json.dumps(["статистический анализ", "визуализация данных"]),
            json.dumps(["Python"]),
        ],
    }
)

CHARACTERISTIC_LABEL_MAP = {
    "experimental": "Экспериментальные навыки",
    "data_analysis": "Анализ данных",
    "computational": "Вычислительные методы",
}


class TestBuildSingleModel:
    """Unit tests for _build_single_model with controlled fixture data."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model = _build_single_model(
            ROLE_ROW, CHARACTERISTIC_ROWS_DF, CHARACTERISTIC_LABEL_MAP
        )

    # -- top-level keys --
    def test_has_all_required_top_level_keys(self):
        expected = {
            "role_id",
            "role_label",
            "top_job_titles",
            "member_count",
            "characteristics",
            "competency_levels",
        }
        assert set(self.model.keys()) == expected

    def test_role_id_is_int(self):
        assert self.model["role_id"] == 5

    def test_role_label_is_string(self):
        assert self.model["role_label"] == "физик-экспериментатор"

    def test_top_job_titles_parsed_from_json(self):
        assert self.model["top_job_titles"] == [
            "физик-экспериментатор",
            "инженер-исследователь",
        ]

    def test_member_count_is_int(self):
        assert self.model["member_count"] == 42

    # -- characteristics --
    def test_characteristics_have_three_entries(self):
        assert len(self.model["characteristics"]) == 3

    def test_each_characteristic_has_required_keys(self):
        required = {"characteristic_id", "label_ru", "proficiency", "top_skills"}
        for char in self.model["characteristics"]:
            assert set(char.keys()) == required

    def test_characteristic_label_ru_uses_label_map(self):
        labels = {
            char["characteristic_id"]: char["label_ru"]
            for char in self.model["characteristics"]
        }
        assert labels["experimental"] == "Экспериментальные навыки"
        assert labels["data_analysis"] == "Анализ данных"
        assert labels["computational"] == "Вычислительные методы"

    def test_top_skills_truncated_to_three(self):
        exp_char = next(
            char for char in self.model["characteristics"]
            if char["characteristic_id"] == "experimental"
        )
        assert len(exp_char["top_skills"]) == 3
        assert exp_char["top_skills"][:3] == [
            "лазерная оптика",
            "интерферометрия",
            "вакуумная техника",
        ]

    # -- competency_levels --
    def test_competency_levels_has_seven_entries(self):
        assert len(self.model["competency_levels"]) == 7

    def test_each_competency_level_has_required_keys(self):
        required = {"label", "weight", "threshold", "characteristics_achieved", "achieved_count"}
        for cl in self.model["competency_levels"]:
            assert set(cl.keys()) == required

    def test_achieved_count_matches_characteristics_achieved_length(self):
        for cl in self.model["competency_levels"]:
            assert cl["achieved_count"] == len(cl["characteristics_achieved"])

    def test_achieved_characteristics_are_monotonic(self):
        """As thresholds rise, the achieved set should shrink (never grow)."""
        achieved_sets = [
            set(cl["characteristics_achieved"])
            for cl in self.model["competency_levels"]
        ]
        for i in range(1, len(achieved_sets)):
            assert achieved_sets[i] <= achieved_sets[i - 1]

    def test_all_characteristics_achieved_at_lowest_level(self):
        """At level 0 (weight 0.10, threshold 1.4), all 3 >= 1.4 → all achieved."""
        cl0 = self.model["competency_levels"][0]
        assert cl0["achieved_count"] == 3
        assert cl0["threshold"] == pytest.approx(1.4)

    def test_only_experimental_achieved_at_high_level(self):
        """At weight 0.8 (threshold 4.2), only experimental proficiency 4.2 >= 4.2 passes."""
        high = [cl for cl in self.model["competency_levels"] if cl["weight"] == 0.80]
        assert len(high) == 1
        assert high[0]["achieved_count"] == 1
        assert "experimental" in high[0]["characteristics_achieved"]

    # -- edge cases --
    def test_empty_characteristic_rows_still_builds(self):
        model = _build_single_model(ROLE_ROW, pd.DataFrame(), CHARACTERISTIC_LABEL_MAP)
        assert model["characteristics"] == []
        assert len(model["competency_levels"]) == 7
        for cl in model["competency_levels"]:
            assert cl["achieved_count"] == 0

    def test_top_skills_fewer_than_three(self):
        """Characteristic with fewer than 3 skills returns all available."""
        comp_char = next(
            char for char in self.model["characteristics"]
            if char["characteristic_id"] == "computational"
        )
        assert len(comp_char["top_skills"]) == 1
        assert comp_char["top_skills"] == ["Python"]

    def test_role_row_with_string_role_id(self):
        """String-cast role_id should be converted to int."""
        row = ROLE_ROW.copy()
        row["role_id"] = "7"
        model = _build_single_model(row, CHARACTERISTIC_ROWS_DF, CHARACTERISTIC_LABEL_MAP)
        assert model["role_id"] == 7

    def test_characteristic_id_missing_from_label_map(self):
        """If characteristic_id not in label_map, label_ru falls back to the id."""
        model = _build_single_model(
            ROLE_ROW,
            CHARACTERISTIC_ROWS_DF,
            {},  # empty map
        )
        for char in model["characteristics"]:
            assert char["label_ru"] == char["characteristic_id"]


# ---------------------------------------------------------------------------
# 5. build_models — extended model JSON + 4 diagrams per role
# ---------------------------------------------------------------------------


class TestBuildModelsExtended:
    """Integration tests for build_models: extended JSON + 4 diagrams per role."""

    def _write_config(
        self, tmp_path: Path, data_dir: Path, models_dir: Path, reports_dir: Path
    ) -> Config:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"""pipeline:
  output_dir: {data_dir}
  models_dir: {models_dir}
  reports_dir: {reports_dir}
characteristics:
  hypotheses:
    - id: domain_knowledge
      label_ru: Доменная база
    - id: experimental
      label_ru: Эксперимент
    - id: data_analysis
      label_ru: Анализ данных
    - id: computational
      label_ru: Вычислительные методы
    - id: professional_texts
      label_ru: Профессиональные тексты
    - id: t_profile
      label_ru: T-профиль
    - id: management
      label_ru: Управление
soft_competences:
  hypotheses:
    - id: thinking
      label_ru: Мышление
    - id: teamwork
      label_ru: Командное Взаимодействие
    - id: leadership
      label_ru: Ответственность и Лидерство
    - id: professional_culture
      label_ru: Профессиональная культура
experience:
  bins: [0, 1, 3, 5, 10]
skills_display:
  top_n: 15
""",
            encoding="utf-8",
        )
        return Config(path=cfg)

    def _setup(self, tmp_path: Path, experience_years: list[float]) -> Config:
        data_dir = tmp_path / "data"
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        data_dir.mkdir(parents=True, exist_ok=True)

        config = self._write_config(tmp_path, data_dir, models_dir, reports_dir)

        roles = pd.DataFrame(
            {
                "role_id": [0, 99],
                "role_label": ["физик-экспериментатор", "шум"],
                "top_titles": [
                    json.dumps(["физик-экспериментатор", "инженер-исследователь"]),
                    json.dumps(["шум"]),
                ],
                "member_count": [3, 1],
                "noise_flag": [False, True],
            }
        )
        roles.to_parquet(data_dir / "roles.parquet", index=False)

        char_ids = [
            "domain_knowledge", "experimental", "data_analysis",
            "computational", "professional_texts", "t_profile", "management",
        ]
        char_scores = pd.DataFrame(
            {
                "role_id": [0] * 7,
                "characteristic_id": char_ids,
                "proficiency": [4.1, 4.8, 3.3, 2.6, 2.7, 2.2, 1.9],
                "top_contributing_skills": [json.dumps(["s1", "s2", "s3"])] * 7,
            }
        )
        char_scores.to_parquet(data_dir / "characteristic_scores.parquet", index=False)

        soft_scores = pd.DataFrame(
            {
                "role_id": [0, 0, 0, 0],
                "soft_id": ["thinking", "teamwork", "leadership", "professional_culture"],
                "proficiency": [4.0, 3.0, 2.0, 3.0],
            }
        )
        soft_scores.to_parquet(data_dir / "soft_scores.parquet", index=False)

        role_archetypes = pd.DataFrame(
            {
                "role_id": [0],
                "archetype": ["Исследователь"],
                "super_fractions": [json.dumps({"Исследователь": 0.8, "Гибрид": 0.2})],
            }
        )
        role_archetypes.to_parquet(data_dir / "role_archetypes.parquet", index=False)

        vacancy_roles = pd.DataFrame(
            {"vacancy_id": ["v0", "v1", "v2"], "role_id": [0, 0, 0]}
        )
        vacancy_roles.to_parquet(data_dir / "vacancy_roles.parquet", index=False)

        # Per-vacancy experience: one row per STEM vacancy.
        vacancy_experience = pd.DataFrame(
            {"vacancy_id": ["v0", "v1", "v2"], "experience_years": experience_years}
        )
        vacancy_experience.to_parquet(data_dir / "vacancy_experience.parquet", index=False)

        skills_per_role = pd.DataFrame(
            {
                "role_id": [0, 0, 0],
                "skill_canonical_name": ["лазерная оптика", "интерферометрия", "вакуумная техника"],
                "tfidf_weight": [0.9, 0.6, 0.3],
            }
        )
        skills_per_role.to_parquet(data_dir / "skills_per_role.parquet", index=False)

        skill_char = pd.DataFrame(
            {
                "skill_canonical_name": ["лазерная оптика", "лазерная оптика", "интерферометрия"],
                "characteristic_id": ["experimental", "computational", "experimental"],
                "nli_score": [0.9, 0.4, 0.8],
            }
        )
        skill_char.to_parquet(data_dir / "skill_characteristic_scores.parquet", index=False)

        return config

    def test_build_models_writes_extended_json_and_four_diagrams(self, tmp_path: Path):
        config = self._setup(
            tmp_path, experience_years=[3.0, 1.5, float("nan")]
        )
        models = build_models(config)

        assert len(models) == 1
        model_path = config.models_dir / "0.json"
        assert model_path.exists()
        assert not (config.models_dir / "99.json").exists()

        model = json.loads(model_path.read_text(encoding="utf-8"))

        assert len(model["characteristics"]) == 7
        for char in model["characteristics"]:
            assert set(char.keys()) == {
                "characteristic_id", "label_ru", "proficiency", "top_skills",
            }

        assert len(model["soft_competences"]) == 4
        for soft in model["soft_competences"]:
            assert set(soft.keys()) == {"soft_id", "label_ru", "proficiency"}

        assert model["archetype"] == "Исследователь"

        exp = model["experience"]
        assert exp["n_not_specified"] == 1
        assert exp["n_with_experience"] == 2
        assert exp["median"] == pytest.approx(2.25)
        assert isinstance(exp["bins"], list) and len(exp["bins"]) == 5

        skills = model["skills"]
        assert len(skills) == 3
        assert skills[0]["skill"] == "лазерная оптика"
        assert skills[0]["tfidf_weight"] == pytest.approx(0.9)
        assert skills[0]["axis_weights"] == {"experimental": 0.9, "computational": 0.4}

        for kind in ("hard", "soft", "experience", "skills"):
            png = config.reports_dir / kind / "0.png"
            assert png.exists(), f"missing {kind} diagram"
            assert png.stat().st_size > 0

    def test_build_models_all_nan_experience(self, tmp_path: Path):
        config = self._setup(tmp_path, experience_years=[float("nan")] * 3)
        models = build_models(config)

        assert len(models) == 1
        model = json.loads((config.models_dir / "0.json").read_text(encoding="utf-8"))
        exp = model["experience"]
        assert exp["n_not_specified"] == 3
        assert exp["n_with_experience"] == 0
        assert exp["median"] is None
