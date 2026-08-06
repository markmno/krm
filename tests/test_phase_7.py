"""Tests for phase_7_validate.py — validation and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from krm.phase_7_validate import (
    _build_markdown_report,
    _validate_phase_2_classification,
    _validate_phase_3_roles,
    _validate_phase_4_skills,
    _validate_phase_5_axes,
    _validate_phase_6_models,
    _validate_integration,
)


# ---------------------------------------------------------------------------
# Helper: lightweight mock Config for tests that only need a few path attrs
# ---------------------------------------------------------------------------

class _FakeConfig:
    """A Config-alike with selectable path properties for unit testing."""

    def __init__(self, **paths: Path | None) -> None:
        self._paths = paths

    @property
    def classified_path(self) -> Path:
        return self._paths["classified_path"]  # type: ignore[index]

    @property
    def roles_path(self) -> Path:
        return self._paths["roles_path"]  # type: ignore[index]

    @property
    def skills_per_role_path(self) -> Path:
        return self._paths["skills_per_role_path"]  # type: ignore[index]

    @property
    def axis_scores_path(self) -> Path:
        return self._paths["axis_scores_path"]  # type: ignore[index]

    @property
    def models_dir(self) -> Path:
        return self._paths["models_dir"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Helpers: build report dicts for _build_markdown_report tests
# ---------------------------------------------------------------------------

def _make_report(phases: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline": "KRM",
        "version": "0.2.0",
        "timestamp": "2026-08-06T00:00:00+00:00",
        "phases": phases,
    }


def _make_phase_data(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


# ===========================================================================
# _build_markdown_report
# ===========================================================================

class TestBuildMarkdownReport:
    """Tests for _build_markdown_report — pure function, no I/O."""

    def test_basic_report_structure(self):
        phases = {
            "phase_2_classification": _make_phase_data(
                total_classified=42,
                missing_columns=[],
                category_distribution={"STEM_RESEARCH": 30, "PURE_IT": 12},
            ),
            "phase_6_models": _make_phase_data(
                total_model_files=5, valid=5, invalid=0, errors=[],
            ),
        }
        report = _make_report(phases)
        md = _build_markdown_report(report)

        assert md.startswith("# KRM Validation Report")
        assert "**Pipeline:** KRM v0.2.0" in md
        assert "**Timestamp:** 2026-08-06T00:00:00+00:00" in md

        assert "## Phase 2 Classification" in md
        assert "**total_classified**: 42" in md
        assert "**missing_columns**: []" in md

        assert "## Phase 6 Models" in md
        assert "**total_model_files**: 5" in md
        assert "**valid**: 5" in md
        assert "**invalid**: 0" in md

    def test_all_phases_present(self):
        phases = {
            f"phase_{i}_dummy": _make_phase_data(result=f"ok_{i}")
            for i in range(6)
        }
        report = _make_report(phases)
        md = _build_markdown_report(report)
        for i in range(6):
            assert f"## Phase {i} Dummy" in md
            assert f"**result**: ok_{i}" in md

    def test_empty_phases_dict(self):
        report = _make_report({})
        md = _build_markdown_report(report)
        assert md.startswith("# KRM Validation Report")
        assert "---" in md

    def test_phase_with_string_value_not_dict(self):
        """Non-dict phase data should not crash iteration."""
        phases = {"phase_broken": "just a string, not a dict"}
        report = _make_report(phases)
        md = _build_markdown_report(report)
        assert "# KRM Validation Report" in md
        assert "## Phase Broken" in md

    def test_skips_underscore_keys(self):
        phases = {
            "phase_test": _make_phase_data(
                visible_key="shown",
                _hidden_key="not_shown",
            ),
        }
        report = _make_report(phases)
        md = _build_markdown_report(report)
        assert "**visible_key**: shown" in md
        assert "_hidden_key" not in md

    def test_int_float_bool_values_formatted(self):
        phases = {
            "phase_numbers": _make_phase_data(
                count=42, score=3.14, flag=True, label="test",
            ),
        }
        report = _make_report(phases)
        md = _build_markdown_report(report)
        assert "**count**: 42" in md
        assert "**score**: 3.14" in md
        assert "**flag**: True" in md

    def test_missing_pipeline_key_raises(self):
        bad_report: dict[str, Any] = {
            "version": "1.0", "timestamp": "...", "phases": {},
        }
        with pytest.raises(KeyError):
            _build_markdown_report(bad_report)

    def test_missing_phases_key_raises(self):
        bad_report: dict[str, Any] = {
            "pipeline": "X", "version": "1.0", "timestamp": "...",
        }
        with pytest.raises(KeyError):
            _build_markdown_report(bad_report)


# ===========================================================================
# _validate_phase_2_classification
# ===========================================================================

class TestValidatePhase2Classification:
    """Tests for _validate_phase_2_classification using temp parquet files."""

    def test_valid_classified_parquet(self, tmp_path: Path):
        df = pd.DataFrame({
            "vacancy_id": ["v001", "v002", "v003", "v004"],
            "title": ["a", "b", "c", "d"],
            "description": ["d1", "d2", "d3", "d4"],
            "stem_category": [
                "STEM_RESEARCH", "STEM_RESEARCH", "PURE_IT", "NON_STEM",
            ],
        })
        parquet_path = tmp_path / "classified.parquet"
        df.to_parquet(parquet_path)

        cfg = _FakeConfig(classified_path=parquet_path)
        result = _validate_phase_2_classification(cfg)  # type: ignore[arg-type]

        assert result["total_classified"] == 4
        assert result["missing_columns"] == []
        assert result["category_distribution"] == {
            "STEM_RESEARCH": 2, "PURE_IT": 1, "NON_STEM": 1,
        }
        assert "f1_target" in result

    def test_missing_columns(self, tmp_path: Path):
        df = pd.DataFrame({
            "vacancy_id": ["v001"],
            "title": ["x"],
        })
        parquet_path = tmp_path / "classified.parquet"
        df.to_parquet(parquet_path)

        cfg = _FakeConfig(classified_path=parquet_path)
        result = _validate_phase_2_classification(cfg)  # type: ignore[arg-type]

        assert result["total_classified"] == 1
        assert "description" in result["missing_columns"]
        assert "stem_category" in result["missing_columns"]
        assert result["category_distribution"] == {}

    def test_empty_dataframe(self, tmp_path: Path):
        df = pd.DataFrame(columns=[
            "vacancy_id", "title", "description", "stem_category",
        ])
        parquet_path = tmp_path / "classified.parquet"
        df.to_parquet(parquet_path)

        cfg = _FakeConfig(classified_path=parquet_path)
        result = _validate_phase_2_classification(cfg)  # type: ignore[arg-type]

        assert result["total_classified"] == 0
        assert result["missing_columns"] == []
        assert result["category_distribution"] == {}

    def test_file_not_found(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.parquet"
        cfg = _FakeConfig(classified_path=missing)
        result = _validate_phase_2_classification(cfg)  # type: ignore[arg-type]

        assert result["status"] == "skipped"
        assert "not found" in result["reason"]

    def test_single_vacancy(self, tmp_path: Path):
        df = pd.DataFrame({
            "vacancy_id": ["v_single"],
            "title": ["researcher"],
            "description": ["physics research"],
            "stem_category": ["STEM_RESEARCH"],
        })
        parquet_path = tmp_path / "classified.parquet"
        df.to_parquet(parquet_path)

        cfg = _FakeConfig(classified_path=parquet_path)
        result = _validate_phase_2_classification(cfg)  # type: ignore[arg-type]

        assert result["total_classified"] == 1
        assert result["category_distribution"] == {"STEM_RESEARCH": 1}


# ===========================================================================
# _validate_phase_6_models
# ===========================================================================

class TestValidatePhase6Models:
    """Tests for _validate_phase_6_models using temp JSON directories."""

    @staticmethod
    def _write_json(dir_path: Path, name: str, data: Any) -> Path:
        fp = dir_path / name
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return fp

    def test_all_valid_models(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for i in range(3):
            self._write_json(models_dir, f"role_{i}.json", {
                "role_id": i, "role_label": f"role_{i}", "axes": [],
            })

        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["total_model_files"] == 3
        assert result["valid"] == 3
        assert result["invalid"] == 0
        assert result["errors"] == []

    def test_mix_valid_and_invalid(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        self._write_json(models_dir, "valid.json", {
            "role_id": 0, "role_label": "ok", "axes": [],
        })
        self._write_json(models_dir, "missing_keys.json", {
            "role_id": 1,
        })
        (models_dir / "malformed.json").write_text("{bad", encoding="utf-8")

        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["total_model_files"] == 3
        assert result["valid"] == 1
        assert result["invalid"] == 2
        assert len(result["errors"]) == 2
        err_names = [e.split(":")[0] for e in result["errors"]]
        assert "missing_keys.json" in err_names
        assert "malformed.json" in err_names

    def test_missing_keys(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        self._write_json(models_dir, "no_role_id.json", {
            "role_label": "test", "axes": [],
        })
        self._write_json(models_dir, "no_axes.json", {
            "role_id": 1, "role_label": "test",
        })

        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["valid"] == 0
        assert result["invalid"] == 2
        assert "no_role_id.json" in str(result["errors"])
        assert "no_axes.json" in str(result["errors"])

    def test_no_models_dir(self, tmp_path: Path):
        missing_dir = tmp_path / "nonexistent"
        cfg = _FakeConfig(models_dir=missing_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["status"] == "skipped"
        assert "not found" in result["reason"]

    def test_empty_models_dir(self, tmp_path: Path):
        models_dir = tmp_path / "empty"
        models_dir.mkdir()
        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["total_model_files"] == 0
        assert result["valid"] == 0
        assert result["invalid"] == 0

    def test_utf8_json_with_cyrillic(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        self._write_json(models_dir, "role_0.json", {
            "role_id": 0,
            "role_label": "физик-экспериментатор",
            "axes": [{"axis_id": "exp", "proficiency": 4.5}],
        })

        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["total_model_files"] == 1
        assert result["valid"] == 1
        assert result["invalid"] == 0

    def test_json_decode_error(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "broken.json").write_text("{bad json", encoding="utf-8")

        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["invalid"] == 1
        assert "broken.json" in result["errors"][0]

    def test_errors_capped_at_10(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for i in range(15):
            (models_dir / f"broken_{i}.json").write_text("{bad", encoding="utf-8")

        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]

        assert result["total_model_files"] == 15
        assert result["invalid"] == 15
        assert len(result["errors"]) == 10

    def test_integrity_target_present(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        self._write_json(models_dir, "r.json", {
            "role_id": 0, "role_label": "x", "axes": [],
        })
        cfg = _FakeConfig(models_dir=models_dir)
        result = _validate_phase_6_models(cfg)  # type: ignore[arg-type]
        assert "integrity_target" in result


# ===========================================================================
# _validate_phase_3_roles
# ===========================================================================

class TestValidatePhase3Roles:
    """Tests for _validate_phase_3_roles with centroid embeddings in parquet."""

    def test_valid_centroids(self, tmp_path: Path):
        c1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        c2 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        c3 = np.array([0.9, 0.1, 0.2, 0.3], dtype=np.float32)
        df = pd.DataFrame({
            "centroid_embedding": [c1.tobytes(), c2.tobytes(), c3.tobytes()],
            "noise_flag": [False, False, True],
            "member_count": [5, 3, 1],
        })
        parquet_path = tmp_path / "roles.parquet"
        df.to_parquet(parquet_path)

        cfg = _FakeConfig(roles_path=parquet_path)
        result = _validate_phase_3_roles(cfg)  # type: ignore[arg-type]

        assert result["total_roles"] == 3
        assert result["noise_roles"] == 1
        assert result["mean_member_count"] == 3.0
        assert "inter_cluster_separation" in result
        assert isinstance(result["inter_cluster_separation"], float)

    def test_too_few_centroids(self, tmp_path: Path):
        c1 = np.array([0.1, 0.2], dtype=np.float32)
        df = pd.DataFrame({
            "centroid_embedding": [c1.tobytes()],
            "noise_flag": [False],
            "member_count": [10],
        })
        (tmp_path / "roles.parquet").write_bytes(
            df.to_parquet(None) if False else b""
        )
        df.to_parquet(tmp_path / "roles.parquet")

        cfg = _FakeConfig(roles_path=tmp_path / "roles.parquet")
        result = _validate_phase_3_roles(cfg)  # type: ignore[arg-type]
        assert "error" in result

    def test_no_centroid_column(self, tmp_path: Path):
        df = pd.DataFrame({"noise_flag": [False], "member_count": [5]})
        df.to_parquet(tmp_path / "roles.parquet")

        cfg = _FakeConfig(roles_path=tmp_path / "roles.parquet")
        result = _validate_phase_3_roles(cfg)  # type: ignore[arg-type]
        assert "error" in result

    def test_file_not_found(self, tmp_path: Path):
        cfg = _FakeConfig(roles_path=tmp_path / "gone.parquet")
        result = _validate_phase_3_roles(cfg)  # type: ignore[arg-type]
        assert result["status"] == "skipped"


# ===========================================================================
# _validate_phase_4_skills
# ===========================================================================

class TestValidatePhase4Skills:
    """Tests for _validate_phase_4_skills using temp parquet files."""

    def test_valid_skills_parquet(self, tmp_path: Path):
        df = pd.DataFrame({
            "role_id": [0, 0, 1, 2],
            "skill_canonical_name": ["a", "b", "c", "d"],
        })
        df.to_parquet(tmp_path / "skills_per_role.parquet")

        cfg = _FakeConfig(skills_per_role_path=tmp_path / "skills_per_role.parquet")
        result = _validate_phase_4_skills(cfg)  # type: ignore[arg-type]

        assert result["total_role_skill_pairs"] == 4
        assert result["roles_with_skills"] == 3
        # source rounds to 1 decimal: float(round(4/3, 1))
        assert result["mean_skills_per_role"] == pytest.approx(1.3)
        assert "precision_at_10_target" in result

    def test_file_not_found(self, tmp_path: Path):
        cfg = _FakeConfig(skills_per_role_path=tmp_path / "nope.parquet")
        result = _validate_phase_4_skills(cfg)  # type: ignore[arg-type]
        assert result["status"] == "skipped"

    def test_uncatalogued_flag(self, tmp_path: Path):
        df = pd.DataFrame({
            "role_id": [0, 1],
            "skill_canonical_name": ["a", "b"],
            "uncatalogued_skills": [True, False],
        })
        df.to_parquet(tmp_path / "skills_per_role.parquet")

        cfg = _FakeConfig(skills_per_role_path=tmp_path / "skills_per_role.parquet")
        result = _validate_phase_4_skills(cfg)  # type: ignore[arg-type]
        assert result["has_uncatalogued"] is True


# ===========================================================================
# _validate_phase_5_axes
# ===========================================================================

class TestValidatePhase5Axes:
    """Tests for _validate_phase_5_axes using temp parquet files."""

    def test_valid_axis_scores(self, tmp_path: Path):
        df = pd.DataFrame({
            "role_id": [0, 0, 1, 1],
            "axis_id": ["exp", "domain", "exp", "domain"],
            "proficiency": [1.0, 2.5, 3.0, 4.5],
        })
        df.to_parquet(tmp_path / "axis_scores.parquet")

        cfg = _FakeConfig(axis_scores_path=tmp_path / "axis_scores.parquet")
        result = _validate_phase_5_axes(cfg)  # type: ignore[arg-type]

        assert result["total_axis_scores"] == 4
        assert result["roles_covered"] == 2
        assert result["axes_covered"] == 2
        assert result["proficiency_range"] == [1.0, 4.5]
        assert result["proficiency_mean"] == pytest.approx(2.75)
        assert "spearman_target" in result

    def test_no_proficiency_column(self, tmp_path: Path):
        df = pd.DataFrame({"role_id": [0], "axis_id": ["exp"]})
        df.to_parquet(tmp_path / "axis_scores.parquet")

        cfg = _FakeConfig(axis_scores_path=tmp_path / "axis_scores.parquet")
        result = _validate_phase_5_axes(cfg)  # type: ignore[arg-type]
        assert "error" in result

    def test_empty_proficiency(self, tmp_path: Path):
        df = pd.DataFrame({
            "role_id": [0], "axis_id": ["exp"], "proficiency": [None],
        })
        df.to_parquet(tmp_path / "axis_scores.parquet")

        cfg = _FakeConfig(axis_scores_path=tmp_path / "axis_scores.parquet")
        result = _validate_phase_5_axes(cfg)  # type: ignore[arg-type]
        assert "error" in result

    def test_file_not_found(self, tmp_path: Path):
        cfg = _FakeConfig(axis_scores_path=tmp_path / "nada.parquet")
        result = _validate_phase_5_axes(cfg)  # type: ignore[arg-type]
        assert result["status"] == "skipped"


# ===========================================================================
# _validate_integration
# ===========================================================================

class TestValidateIntegration:
    """Tests for _validate_integration — cross-phase coverage checks."""

    def test_all_phases_present(self, tmp_path: Path):
        classified = pd.DataFrame({
            "vacancy_id": ["v1", "v2", "v3", "v4", "v5"],
            "stem_category": [
                "STEM_RESEARCH", "STEM_RESEARCH", "PURE_IT",
                "NON_STEM", "STEM_RESEARCH",
            ],
        })
        classified.to_parquet(tmp_path / "classified.parquet")

        roles = pd.DataFrame({
            "noise_flag": [False, False, False, True],
        })
        roles.to_parquet(tmp_path / "roles.parquet")

        skills = pd.DataFrame({
            "role_id": [0, 0, 1, 2, 3],
        })
        skills.to_parquet(tmp_path / "skills_per_role.parquet")

        axes = pd.DataFrame({
            "role_id": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "axis_id": [
                "a", "b", "c", "d", "e", "f",
                "a", "b", "c", "d", "e", "f",
            ],
        })
        axes.to_parquet(tmp_path / "axis_scores.parquet")

        cfg = _FakeConfig(
            classified_path=tmp_path / "classified.parquet",
            roles_path=tmp_path / "roles.parquet",
            skills_per_role_path=tmp_path / "skills_per_role.parquet",
            axis_scores_path=tmp_path / "axis_scores.parquet",
        )
        result = _validate_integration(cfg)  # type: ignore[arg-type]

        assert result["total_vacancies"] == 5
        assert result["stem_research_vacancies"] == 3
        assert result["total_roles"] == 4
        assert result["active_roles"] == 3
        assert result["roles_with_skills"] == 4
        assert result["roles_with_axis_scores"] == 2
        assert result["roles_with_complete_axes"] == 2  # both roles have all 6 axes

    def test_no_files(self, tmp_path: Path):
        cfg = _FakeConfig(
            classified_path=tmp_path / "c.parquet",
            roles_path=tmp_path / "r.parquet",
            skills_per_role_path=tmp_path / "s.parquet",
            axis_scores_path=tmp_path / "a.parquet",
        )
        result = _validate_integration(cfg)  # type: ignore[arg-type]
        assert result == {}

    def test_partial_data(self, tmp_path: Path):
        classified = pd.DataFrame({
            "vacancy_id": ["v1"],
            "stem_category": ["STEM_RESEARCH"],
        })
        classified.to_parquet(tmp_path / "classified.parquet")

        cfg = _FakeConfig(
            classified_path=tmp_path / "classified.parquet",
            roles_path=tmp_path / "r.parquet",
            skills_per_role_path=tmp_path / "s.parquet",
            axis_scores_path=tmp_path / "a.parquet",
        )
        result = _validate_integration(cfg)  # type: ignore[arg-type]
        assert result["total_vacancies"] == 1
        assert result["stem_research_vacancies"] == 1
        assert "total_roles" not in result
