"""Tests for krm.visualization — the эталон renderers.

Locks the observable contract of the renderers (a valid PNG at the requested
path, and sane edge behavior) and the pure helper functions, without depending
on any specific pixel content.
"""

from __future__ import annotations

import pytest

from krm.visualization import (
    _SHORT_LABELS,
    _skill_label,
    _wrap,
    compute_experience_bins,
    render_career_heatmap,
    render_career_trajectory,
    render_comparison,
    render_portfolio,
    render_role_card,
    render_role_experience,
    render_role_skills,
    render_role_soft_spider,
    render_role_spider,
    render_summary,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _model(role_id: int = 0, label: str = "Физик-экспериментатор") -> dict:
    characteristics = [
        {"characteristic_id": cid, "label_ru": lbl, "proficiency": prof, "top_skills": sk}
        for cid, lbl, prof, sk in [
            ("domain_knowledge", "Доменная база", 4.1, ["квантовая механика"]),
            ("experimental", "Эксперимент", 4.8, ["лазерная оптика", "интерферометрия"]),
            ("data_analysis", "Анализ данных", 3.3, ["статистическая обработка"]),
            ("computational", "Вычислительные методы", 2.6, ["Python"]),
            ("professional_texts", "Профессиональные тексты", 2.7, ["научные публикации"]),
            ("t_profile", "T-профиль", 3.0, ["междисциплинарные исследования"]),
            ("management", "Управление", 1.9, ["руководство лабораторией"]),
        ]
    ]
    prof_map = {c["characteristic_id"]: c["proficiency"] for c in characteristics}
    scale = [("Бакалавр", 0.10), ("Магистр", 0.25), ("Аспирант / МНС", 0.40),
             ("Научный сотрудник", 0.60), ("Старший научный сотрудник", 0.80),
             ("Ведущий научный сотрудник", 0.95), ("Главный научный сотрудник", 1.00)]
    levels = []
    for lvl_label, weight in scale:
        threshold = 1.0 + 4.0 * weight
        achieved = [cid for cid, p in prof_map.items() if p >= threshold]
        levels.append({
            "label": lvl_label, "weight": weight, "threshold": round(threshold, 2),
            "characteristics_achieved": achieved, "achieved_count": len(achieved),
        })
    return {
        "role_id": role_id,
        "role_label": label,
        "top_job_titles": ["физик-экспериментатор", "инженер-исследователь"],
        "member_count": 128,
        "characteristics": characteristics,
        "competency_levels": levels,
    }


_ORDER = ["domain_knowledge", "experimental", "data_analysis", "computational",
          "professional_texts", "t_profile", "management"]


def _assert_png(path) -> None:
    assert path.exists()
    data = path.read_bytes()
    assert data[:8] == _PNG_MAGIC
    assert len(data) > 10_000


class TestSkillLabel:
    def test_dict_form_extracts_skill(self):
        assert _skill_label({"skill": "лазерная оптика"}) == "лазерная оптика"

    def test_str_form_passthrough(self):
        assert _skill_label("Python") == "Python"

    def test_dict_with_name_fallback(self):
        assert _skill_label({"name": "R"}) == "R"


class TestWrap:
    def test_truncates_to_max_lines(self):
        lines = _wrap("один два три четыре пять", width=8, max_lines=2)
        assert len(lines) == 2
        assert lines[-1].endswith("…")

    def test_empty_returns_empty(self):
        assert _wrap("", width=10) == []

    def test_short_text_single_line(self):
        assert _wrap("Python", width=10) == ["Python"]


class TestRenderers:
    def test_render_role_spider_writes_valid_png(self, tmp_path):
        out = render_role_spider(_model(), _ORDER, tmp_path / "spider.png")
        _assert_png(out)

    def test_render_role_card_writes_valid_png(self, tmp_path):
        out = render_role_card(_model(), _ORDER, tmp_path / "card.png")
        _assert_png(out)

    def test_render_portfolio_writes_valid_png(self, tmp_path):
        models = [_model(0, "Физик-экспериментатор"), _model(1, "Химик-исследователь")]
        out = render_portfolio(models, _ORDER, tmp_path / "portfolio.png")
        _assert_png(out)

    def test_render_spider_empty_characteristics_raises(self, tmp_path):
        model = _model()
        model["characteristics"] = []
        with pytest.raises(ValueError):
            render_role_spider(model, _ORDER, tmp_path / "empty.png")

    def test_render_portfolio_empty_raises(self, tmp_path):
        with pytest.raises(ValueError):
            render_portfolio([], _ORDER, tmp_path / "none.png")


_AXIS_LABELS = {
    "domain_knowledge": "Доменная база",
    "experimental": "Эксперимент",
    "data_analysis": "Анализ данных",
    "computational": "Вычислительные методы",
    "professional_texts": "Профессиональные тексты",
    "t_profile": "T-профиль",
    "management": "Управление",
}


def _career_levels() -> list[dict]:
    """A junior→C-level trajectory with the technical-peak-then-decline shape."""
    anchors = {
        0: {"domain_knowledge": 2.0, "experimental": 3.2, "data_analysis": 1.8,
            "computational": 2.2, "professional_texts": 1.0, "t_profile": 1.5,
            "management": 1.0},
        2: {"domain_knowledge": 3.6, "experimental": 4.8, "data_analysis": 3.0,
            "computational": 2.8, "professional_texts": 2.0, "t_profile": 2.0,
            "management": 1.8},
        6: {"domain_knowledge": 3.4, "experimental": 1.8, "data_analysis": 2.2,
            "computational": 1.4, "professional_texts": 4.8, "t_profile": 3.0,
            "management": 5.0},
    }
    names = ["Младший специалист", "Специалист", "Старший специалист",
             "Ведущий специалист", "Главный специалист",
             "Руководитель направления", "Топ-менеджмент (C-level)"]
    levels = []
    for i in range(7):
        if i in anchors:
            axes = anchors[i]
        elif i < 2:
            t = i / 2
            axes = {k: anchors[0][k] + (anchors[2][k] - anchors[0][k]) * t for k in anchors[0]}
        else:
            t = (i - 2) / 4
            axes = {k: anchors[2][k] + (anchors[6][k] - anchors[2][k]) * t for k in anchors[0]}
        levels.append({"label": names[i], "axes": axes})
    return levels


class TestCareerRenderers:
    def test_render_career_trajectory_writes_valid_png(self, tmp_path):
        out = render_career_trajectory(
            _career_levels(), _ORDER, _AXIS_LABELS, tmp_path / "traj.png"
        )
        _assert_png(out)

    def test_render_career_heatmap_writes_valid_png(self, tmp_path):
        out = render_career_heatmap(
            _career_levels(), _ORDER, _AXIS_LABELS, tmp_path / "heat.png"
        )
        _assert_png(out)

    def test_render_comparison_writes_valid_png(self, tmp_path):
        models = [_model(0, "Физик-экспериментатор"), _model(1, "Химик-исследователь")]
        out = render_comparison(models, _ORDER, tmp_path / "cmp.png")
        _assert_png(out)

    def test_render_summary_writes_valid_png(self, tmp_path):
        out = render_summary(
            ["Физик-экспериментатор", "Химик-исследователь"],
            [284, 471],
            [38, 41],
            tmp_path / "summary.png",
        )
        _assert_png(out)


class TestShortLabels:
    def test_seven_axes(self):
        assert set(_SHORT_LABELS) == {
            "domain_knowledge",
            "experimental",
            "data_analysis",
            "computational",
            "professional_texts",
            "t_profile",
            "management",
        }

    def test_renamed_labels(self):
        assert _SHORT_LABELS["domain_knowledge"] == "Домен"
        assert _SHORT_LABELS["professional_texts"] == "Тексты"
        assert _SHORT_LABELS["t_profile"] == "T-профиль"


class TestExperienceBins:
    def test_histogram_median_and_nan_count(self):
        years = [0.5, 2.0, 4.0, 7.0, 15.0, float("nan"), float("nan")]
        labelled, median, n_not_specified = compute_experience_bins(years, [0, 1, 3, 5, 10])
        assert [r for r, _ in labelled] == ["0–1", "1–3", "3–5", "5–10", "10+"]
        assert [c for _, c in labelled] == [1, 1, 1, 1, 1]
        assert median == pytest.approx(4.0)
        assert n_not_specified == 2

    def test_all_nan_yields_none_median(self):
        labelled, median, n_not_specified = compute_experience_bins(
            [float("nan"), float("nan")], [0, 1, 3, 5, 10]
        )
        assert median is None
        assert n_not_specified == 2
        assert all(c == 0 for _, c in labelled)

    def test_empty_years(self):
        labelled, median, n_not_specified = compute_experience_bins([], [0, 1, 3, 5, 10])
        assert median is None
        assert n_not_specified == 0
        assert all(c == 0 for _, c in labelled)


class TestExperienceRenderer:
    def test_render_writes_valid_png(self, tmp_path):
        years = [0.5, 1.0, 2.0, 4.0, 7.0, 15.0, float("nan")]
        out = render_role_experience(
            "Физик-экспериментатор", years, tmp_path / "exp.png", bins=[0, 1, 3, 5, 10]
        )
        _assert_png(out)

    def test_render_empty_renders_without_raising(self, tmp_path):
        out = render_role_experience(
            "Физик-экспериментатор", [], tmp_path / "exp_empty.png", bins=[0, 1, 3, 5, 10]
        )
        assert out.exists()
        assert out.read_bytes()[:8] == _PNG_MAGIC


_SKILL_AXIS_LABELS = {
    "domain_knowledge": "Доменная база",
    "experimental": "Эксперимент",
    "data_analysis": "Анализ данных",
    "computational": "Вычислительные методы",
    "professional_texts": "Профессиональные тексты",
    "t_profile": "T-профиль",
    "management": "Управление",
}


class TestSkillsRenderer:
    def test_render_writes_valid_png(self, tmp_path):
        skills = [
            {"skill": "Python", "tfidf_weight": 0.9,
             "axis_weights": {"computational": 0.8, "data_analysis": 0.4}},
            {"skill": "статистика", "tfidf_weight": 0.7,
             "axis_weights": {"data_analysis": 0.9}},
            {"skill": "научные публикации", "tfidf_weight": 0.5,
             "axis_weights": {"professional_texts": 0.7}},
        ]
        out = render_role_skills(
            "Физик-экспериментатор", skills, _SKILL_AXIS_LABELS, tmp_path / "skills.png"
        )
        _assert_png(out)


class TestSoftSpiderRenderer:
    def test_render_writes_valid_png(self, tmp_path):
        model = _model()
        model["soft_competences"] = [
            {"soft_id": "thinking", "label_ru": "Мышление", "proficiency": 4.2},
            {"soft_id": "teamwork", "label_ru": "Командное Взаимодействие", "proficiency": 3.5},
            {"soft_id": "leadership", "label_ru": "Ответственность и Лидерство", "proficiency": 2.8},
            {"soft_id": "professional_culture", "label_ru": "Профессиональная культура", "proficiency": 3.9},
        ]
        soft_order = ["thinking", "teamwork", "leadership", "professional_culture"]
        out = render_role_soft_spider(model, soft_order, tmp_path / "soft.png")
        _assert_png(out)
