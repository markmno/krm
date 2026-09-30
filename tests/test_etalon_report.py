"""Tests for scripts.generate_etalon — mock data coherence and trajectory logic.

Locks the internal consistency of the mock эталон data (KPI sums reconcile,
dominant pairs are valid) and the junior→C-level trajectory invariants
(monotonic non-decreasing axes, non-linear per-axis growth, dominant pair ahead
of the rest at C-level, no axis at the 1.0 floor for non-dominant).
"""

from __future__ import annotations

from scripts.generate_etalon import (
    AXIS_IDS,
    CAREER_LEVELS,
    GRADES,
    GROWTH_SHAPE,
    KPIS,
    LEADERSHIP_TARGET,
    REFERENCES,
    ROLES,
    STAGE_REQUIREMENTS,
    STAGE_TEAMS,
    TRL_STAGES,
    _best_switch,
    _gap_score,
    _grow,
    _leadership_edges,
    _shared_axis,
    _stage_coverage,
    _trajectory,
    _transitions,
)
from scripts.generate_report import JUNIOR_FRACTION, _build_roles


def _kpi(keyword: str) -> str:
    return next(v for v, label in KPIS if keyword in label)


class TestMockDataCoherence:
    def test_member_counts_reconcile_with_coverage(self):
        total_members = sum(r["member_count"] for r in ROLES)
        stem = int(_kpi("STEM").replace("\u00a0", "").replace(" ", ""))
        coverage_pct = int(_kpi("Покрытие").replace("%", ""))
        assert round(total_members / stem * 100) == coverage_pct

    def test_skill_counts_sum_to_extracted_total(self):
        assert sum(r["skill_count"] for r in ROLES) == int(_kpi("Навыков").replace("\u00a0", ""))

    def test_roles_count_matches_kpi(self):
        assert len(ROLES) == int(_kpi("Ролей"))

    def test_every_role_has_full_axis_coverage(self):
        for r in ROLES:
            assert set(r["junior"]) == set(AXIS_IDS)
            assert set(r["clevel"]) == set(AXIS_IDS)

    def test_exactly_six_roles(self):
        assert len(ROLES) == 6

    def test_each_axis_has_at_least_four_skills(self):
        for r in ROLES:
            for cid in AXIS_IDS:
                assert len(r["skills"].get(cid, [])) >= 4, (r["label"], cid)

    def test_dominant_pair_is_two_valid_axes(self):
        for r in ROLES:
            assert len(r["dominant"]) == 2
            assert set(r["dominant"]) <= set(AXIS_IDS)

    def test_dominant_axes_exceed_nondominant_at_clevel(self):
        for r in ROLES:
            dom = set(r["dominant"])
            dom_min = min(r["clevel"][d] for d in dom)
            nondom_max = max(r["clevel"][a] for a in AXIS_IDS if a not in dom)
            assert dom_min > nondom_max

    def test_nondominant_axes_above_floor(self):
        for r in ROLES:
            dom = set(r["dominant"])
            for a in AXIS_IDS:
                if a not in dom:
                    assert r["clevel"][a] >= 2.0


class TestTrajectory:
    def test_seven_levels_in_order(self):
        assert [l["label"] for l in _trajectory(ROLES[0])] == CAREER_LEVELS

    def test_all_axes_monotonic_non_decreasing(self):
        for role in ROLES:
            levels = _trajectory(role)
            for i in range(len(levels) - 1):
                for axis in AXIS_IDS:
                    assert levels[i + 1]["axes"][axis] >= levels[i]["axes"][axis], (
                        role["label"], axis, i,
                    )

    def test_no_axis_declines_at_clevel(self):
        for role in ROLES:
            levels = _trajectory(role)
            for axis in AXIS_IDS:
                assert levels[-1]["axes"][axis] == max(
                    l["axes"][axis] for l in levels
                )

    def test_dominant_pair_ahead_at_clevel(self):
        for role in ROLES:
            dom = set(role["dominant"])
            clevel = _trajectory(role)[-1]["axes"]
            dom_min = min(clevel[d] for d in dom)
            nondom_max = max(clevel[a] for a in AXIS_IDS if a not in dom)
            assert dom_min > nondom_max, role["label"]


class TestGrow:
    def test_endpoints(self):
        assert _grow(1.0, 3.0, 0.0, "management") == 1.0
        assert _grow(1.0, 3.0, 1.0, "management") == 3.0

    def test_monotonic_for_all_shapes(self):
        for axis, p in GROWTH_SHAPE.items():
            prev = 0.0
            for i in range(8):
                v = _grow(1.0, 5.0, i / 7, axis)
                assert v >= prev, (axis, i)
                prev = v

    def test_management_grows_late(self):
        # p > 1 → at t=0.5 the value is below the linear midpoint (3.0).
        assert _grow(1.0, 5.0, 0.5, "management") < 3.0

    def test_experimental_grows_early(self):
        # p < 1 → at t=0.5 the value is above the linear midpoint (3.0).
        assert _grow(1.0, 5.0, 0.5, "experimental") > 3.0


class TestTransitions:
    def test_every_edge_shares_exactly_one_dominant_axis(self):
        for e in _transitions():
            shared = _shared_axis(e["source"], e["target"])
            assert shared is not None
            assert shared in ROLES[e["source"]]["dominant"]
            assert shared in ROLES[e["target"]]["dominant"]

    def test_edges_have_nonempty_gap(self):
        for e in _transitions():
            assert any(v > 0 for v in e["gap"].values())

    def test_manager_has_minimum_leadership_gap(self):
        edges = _leadership_edges()
        scores = {e["source"]: _gap_score(e["gap"]) for e in edges}
        assert scores[2] == min(scores.values())

    def test_leadership_target_covers_two_axes(self):
        assert set(LEADERSHIP_TARGET) == {"management", "domain_knowledge"}

    def test_every_role_has_a_best_switch(self):
        for r in ROLES:
            best = _best_switch(r["role_id"])
            assert best["target"] != r["role_id"]
            assert _gap_score(best["gap"]) > 0


class TestReferences:
    def test_references_have_urls_and_no_game_analogy(self):
        assert all(url.startswith("http") for _, url, _ in REFERENCES)
        blob = " ".join(n + " " + u + " " + d for n, u, d in REFERENCES)
        assert "D&D" not in blob
        assert "Dungeons" not in blob.lower()


class TestStages:
    def test_five_grades(self):
        assert len(GRADES) == 5

    def test_grades_have_monotonic_representative_t(self):
        ts = [g["t"] for g in GRADES]
        assert ts == sorted(ts)
        assert ts[0] == 0.0 and ts[-1] == 1.0

    def test_grade_levels_partition_career_levels(self):
        levels = [lv for g in GRADES for lv in g["levels"]]
        assert sorted(levels) == list(range(len(CAREER_LEVELS)))

    def test_every_stage_fully_covered(self):
        for s in TRL_STAGES:
            for a in AXIS_IDS:
                supply, req = _stage_coverage(s["id"])[a]
                assert supply >= req - 0.01, (s["id"], a, supply, req)

    def test_every_stage_has_a_team(self):
        for s in TRL_STAGES:
            assert STAGE_TEAMS[s["id"]], s["id"]
            for rid, gid, cnt in STAGE_TEAMS[s["id"]]:
                assert 0 <= rid < len(ROLES)
                assert 0 <= gid < len(GRADES)
                assert cnt >= 1


class TestSevenAxesMigration:
    SEVEN_IDS = [
        "domain_knowledge", "experimental", "data_analysis", "computational",
        "professional_texts", "t_profile", "management",
    ]

    def test_axis_ids_are_seven_in_scope_order(self):
        assert AXIS_IDS == self.SEVEN_IDS

    def test_growth_shape_has_seven_keys(self):
        assert set(GROWTH_SHAPE) == set(self.SEVEN_IDS)
        assert GROWTH_SHAPE["professional_texts"] == 1.6
        assert GROWTH_SHAPE["t_profile"] == 1.0

    def test_junior_fraction_has_seven_keys(self):
        assert set(JUNIOR_FRACTION) == set(self.SEVEN_IDS)
        assert "literature" not in JUNIOR_FRACTION

    def test_stage_requirements_have_seven_axes(self):
        for s in STAGE_REQUIREMENTS:
            assert set(s["axes"]) == set(self.SEVEN_IDS)

    def test_build_roles_tolerates_extended_schema(self):
        model = {
            "role_id": 0,
            "role_label": "Биолог-экспериментатор",
            "top_job_titles": ["биолог-экспериментатор"],
            "member_count": 100,
            "archetype": "Исследователь",
            "characteristics": [
                {"characteristic_id": cid, "label_ru": cid, "proficiency": 3.0,
                 "top_skills": [f"skill_{cid}"]}
                for cid in self.SEVEN_IDS
            ],
            "soft_competences": [
                {"soft_id": sid, "label_ru": sid, "proficiency": 3.0}
                for sid in ("thinking", "teamwork", "leadership", "professional_culture")
            ],
            "experience": {
                "bins": [{"range": "0–1", "count": 5}],
                "median": 2.0, "n_with_experience": 5, "n_not_specified": 1,
            },
            "skills": [
                {"skill": "Python", "tfidf_weight": 0.9,
                 "axis_weights": {"computational": 0.8}},
            ],
            "competency_levels": [],
        }
        roles = _build_roles({0: model}, {0: ["Python", "R"]})
        assert len(roles) == 1
        r = roles[0]
        assert r["source_role_id"] == 0
        assert r["soft_scores"] == model["soft_competences"]
        assert r["experience"] == model["experience"]
        assert r["skills_detail"] == model["skills"]
        assert set(r["skills"]) == set(self.SEVEN_IDS)
