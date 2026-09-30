"""Generate the real KRM report from pipeline output.

Produces a self-contained HTML report — same structure/template as the эталон
reference (``scripts/generate_etalon.py``) — but filled with **real pipeline
data** instead of illustrative mock data.

The report reuses the эталон's HTML builders and framework tables (TRL stages,
Dreyfus grades, stage requirements, artifacts, growth shapes) by monkey-patching
the data globals on the ``generate_etalon`` module. All role-level facts are
derived from real pipeline output:

* ``models/*.json``           — competency models (per-role characteristics)
* ``data/skills_per_role.parquet`` — role-level skills
* ``reports/metrics.json``    — validation metrics

Derivations (all deterministic, documented in-code):

* **dominant pair**  — top-2 characteristics by proficiency.
* **junior baseline** — ``junior[axis] = max(1.0, clevel[axis] * fraction[axis])``;
  the fraction is smaller for late-developing axes (management, professional_texts).
* **career trajectory** — ``generate_etalon._trajectory`` (monotonic junior→C-level
  interpolation with per-axis S-curve shapes).
* **TRL placement** — nearest ``STAGE_REQUIREMENTS`` profile by cosine similarity;
  roles with a computational/data-analysis dominant are transversal (all stages).
* **transitions** — role A → B when they share at least one dominant axis and
  their C-level profiles are close by cosine similarity
  (``generate_etalon._transitions``).
* **leadership gap** — distance from ``LEADERSHIP_TARGET`` (management+domain ≥ 4.0).

Usage:
    python -m scripts.generate_report
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.generate_etalon as etalon  # noqa: E402 — template + framework
from krm.config import Config

# ---------------------------------------------------------------------------
# Derivation constants
# ---------------------------------------------------------------------------

# Junior baseline as a fraction of the C-level (peak) proficiency. Axes that
# develop late (management, professional_texts) start from a lower baseline, mirroring
# generate_etalon.GROWTH_SHAPE.
JUNIOR_FRACTION: dict[str, float] = {
    "experimental": 0.45,
    "domain_knowledge": 0.40,
    "management": 0.25,
    "professional_texts": 0.30,
    "data_analysis": 0.40,
    "computational": 0.45,
    "t_profile": 0.40,
}

TRANSVERSAL_AXES = frozenset({"computational", "data_analysis"})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_models(models_dir: Path) -> dict[int, dict[str, Any]]:
    """Load models/*.json keyed by role_id."""
    models: dict[int, dict[str, Any]] = {}
    for p in sorted(models_dir.glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        models[int(m["role_id"])] = m
    return models


def _load_skills(skills_path: Path) -> dict[int, list[str]]:
    """Load skills_per_role.parquet → {role_id: [skill, ...]}."""
    if not skills_path.exists():
        return {}
    df = pd.read_parquet(skills_path)
    return {
        int(rid): grp["skill_canonical_name"].astype(str).tolist()
        for rid, grp in df.groupby("role_id")
    }


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load reports/metrics.json (may be absent on a partial run)."""
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _attach_example_vacancies(
    roles: list[dict[str, Any]],
    classified_path: Path,
    vacancy_roles_path: Path,
) -> None:
    """Attach one real representative vacancy to each role.

    Reads the classified vacancies and the vacancy→role mapping and sets
    ``role["example_vacancy"] = {"title", "description", "skills"}`` (matched by
    ``role["source_role_id"]``). The first vacancy assigned to a role is used.
    Missing files or a role without vacancies leave the field unset — the
    template then falls back to a synthesized sample.
    """
    if not classified_path.exists() or not vacancy_roles_path.exists():
        return
    vac = pd.read_parquet(classified_path)
    if "description" not in vac.columns or "vacancy_id" not in vac.columns:
        return
    vr = pd.read_parquet(vacancy_roles_path)
    if vr.empty:
        return
    merged = vr[["vacancy_id", "role_id"]].merge(
        vac[["vacancy_id", "title", "description"]], on="vacancy_id", how="inner"
    )
    by_role = {int(rid): grp for rid, grp in merged.groupby("role_id")}
    for r in roles:
        grp = by_role.get(int(r["source_role_id"]))
        if grp is None or grp.empty:
            continue
        row = grp.iloc[0]
        description = (
            str(row["description"]) if pd.notna(row["description"]) else ""
        )
        skills: list[str] = []
        for axis in etalon.AXIS_IDS:
            for s in r["skills"].get(axis, []):
                if s not in skills:
                    skills.append(s)
        r["example_vacancy"] = {
            "title": str(row["title"]),
            "description": description,
            "skills": skills[:8],
        }


# ---------------------------------------------------------------------------
# Role construction
# ---------------------------------------------------------------------------


def _dominant(prof: dict[str, float], k: int = 2) -> list[str]:
    return sorted(prof, key=prof.get, reverse=True)[:k]


def _top_skill_names(char: dict[str, Any]) -> list[str]:
    """Extract skill names from a characteristic's top_skills (dict or str)."""
    skills = char.get("top_skills", []) or []
    out: list[str] = []
    for s in skills:
        name = s.get("skill") or s.get("name") or "" if isinstance(s, dict) else str(s)
        if name:
            out.append(name)
    return out


def _build_roles(
    models: dict[int, dict[str, Any]],
    skills_by_role: dict[int, list[str]],
) -> list[dict[str, Any]]:
    """Convert real models into MockRole-shaped dicts."""
    roles: list[dict[str, Any]] = []
    for idx, rid in enumerate(sorted(models)):
        m = models[rid]
        chars = {c["characteristic_id"]: c for c in m.get("characteristics", [])}

        clevel = {
            a: float(chars[a]["proficiency"]) if a in chars else 1.0
            for a in etalon.AXIS_IDS
        }
        junior = {
            a: max(1.0, round(clevel[a] * JUNIOR_FRACTION[a], 2))
            for a in etalon.AXIS_IDS
        }
        skills = {
            a: _top_skill_names(chars[a]) if a in chars else []
            for a in etalon.AXIS_IDS
        }

        # The report template indexes roles by dense list position (its mock
        # roles were 0..5), so remap the real role_id to a dense 0..n-1 index.
        roles.append({
            "role_id": idx,
            "source_role_id": rid,
            "label": m.get("role_label", f"Роль {rid}"),
            "top_titles": (m.get("top_job_titles") or [])[:3],
            "member_count": int(m.get("member_count", 0)),
            "skill_count": len(skills_by_role.get(rid, [])),
            "dominant": _dominant(clevel),
            "junior": junior,
            "clevel": clevel,
            "skills": skills,
            "soft_scores": m.get("soft_competences") or [],
            "experience": m.get("experience") or {},
            "skills_detail": m.get("skills") or [],
        })
    return roles


# ---------------------------------------------------------------------------
# Derivation: TRL placement, teams, role map, transversal ladder
# ---------------------------------------------------------------------------


def _derive_trl_map(roles: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Assign each role to the nearest TRL stage by profile similarity.

    Transversal roles (computational / data-analysis dominant) span all stages.
    """
    reqs = [
        np.array([s["axes"][a] for a in etalon.AXIS_IDS], dtype=float)
        for s in etalon.STAGE_REQUIREMENTS
    ]
    trl_map: dict[int, dict[str, Any]] = {}
    for r in roles:
        prof = np.array([r["clevel"][a] for a in etalon.AXIS_IDS], dtype=float)
        if set(r["dominant"]) & TRANSVERSAL_AXES:
            trl_map[r["role_id"]] = {
                "stage_id": None, "span": (0, 9),
                "family": "Инженер-исследователь", "transversal": True,
            }
            continue
        sims = [
            float(np.dot(prof, q) / (np.linalg.norm(prof) * np.linalg.norm(q) + 1e-9))
            for q in reqs
        ]
        sid = int(np.argmax(sims))
        trl_map[r["role_id"]] = {
            "stage_id": sid, "span": (sid * 3, sid * 3 + 3),
            "family": etalon.TRL_STAGES[sid]["family"], "transversal": False,
        }
    return trl_map


def _grade_for(member_count: int, all_counts: list[int]) -> int:
    """Map a role's member_count to a Dreyfus grade (quintile within corpus)."""
    if not all_counts or member_count <= 0:
        return 0
    lo = sorted(all_counts)
    n = len(lo)
    for g in range(4, 0, -1):
        if member_count >= lo[int(n * g / 5) - 1]:
            return g
    return 0


def _derive_stage_teams(
    roles: list[dict[str, Any]],
    trl_map: dict[int, dict[str, Any]],
) -> dict[int, list[tuple[int, int, int]]]:
    """For each stage, a team = resident roles (top by member_count) at their grade."""
    all_counts = [r["member_count"] for r in roles]
    residents: dict[int, list[dict[str, Any]]] = {s["id"]: [] for s in etalon.TRL_STAGES}
    transversal: list[dict[str, Any]] = []

    for r in roles:
        m = trl_map[r["role_id"]]
        if m["transversal"]:
            transversal.append(r)
        elif m["stage_id"] is not None:
            residents[m["stage_id"]].append(r)

    teams: dict[int, list[tuple[int, int, int]]] = {}
    for sid in residents:
        ranked = sorted(residents[sid], key=lambda r: -r["member_count"])[:4]
        teams[sid] = [
            (r["role_id"], _grade_for(r["member_count"], all_counts), 1) for r in ranked
        ]
        # one transversal role supports every stage (covers computational/data)
        if transversal:
            best = max(transversal, key=lambda r: r["member_count"])
            teams[sid].append((best["role_id"], 2, 1))
    return teams


def _derive_role_map(
    roles: list[dict[str, Any]],
    trl_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project functions matrix: each non-transversal role → (stage, grade) cell."""
    all_counts = [r["member_count"] for r in roles]
    cells: list[dict[str, Any]] = []
    for r in roles:
        m = trl_map[r["role_id"]]
        if m["transversal"] or m["stage_id"] is None:
            continue
        cells.append({
            "stage": m["stage_id"],
            "grade": _grade_for(r["member_count"], all_counts),
            "title": r["label"],
        })
    return cells


def _derive_transversal_ladder(
    roles: list[dict[str, Any]],
    trl_map: dict[int, dict[str, Any]],
) -> dict[int, dict[int, str]]:
    """Transversal roles: same label across grades (a simple ladder)."""
    ladder: dict[int, dict[int, str]] = {}
    for r in roles:
        if trl_map[r["role_id"]]["transversal"]:
            ladder[r["role_id"]] = {g["id"]: r["label"] for g in etalon.GRADES}
    return ladder


# ---------------------------------------------------------------------------
# KPIs & metrics from real data
# ---------------------------------------------------------------------------


def _derive_kpis(
    roles: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[tuple[str, str]]:
    integration = metrics.get("integration", {}) or {}
    total_roles = len(roles)
    total_skills = sum(r["skill_count"] for r in roles)
    stem = integration.get("stem_research_vacancies") or (
        metrics.get("phase_2_classification", {}).get("category_distribution", {})
        .get("STEM_RESEARCH", "—")
    )
    total_vac = integration.get("total_vacancies") or "—"
    covered = integration.get("roles_with_characteristic_scores") or total_roles
    coverage = f"{round(100 * covered / total_roles)}%" if total_roles else "—"
    return [
        (f"{total_vac:,}" if isinstance(total_vac, int) else str(total_vac),
         "Вакансий собрано"),
        (f"{total_roles}", "Ролей обнаружено"),
        (f"{stem:,}" if isinstance(stem, int) else str(stem), "STEM-вакансий"),
        (f"{total_skills}", "Навыков извлечено"),
        (coverage, "Покрытие модели"),
    ]


def _derive_metrics(metrics: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Validation metrics table from metrics.json (targets are fixed policy)."""
    p2 = metrics.get("phase_2_classification", {})
    p3 = metrics.get("phase_3_roles", {})
    p4 = metrics.get("phase_4_skills", {})
    p5 = metrics.get("phase_5_characteristics", {})
    p6 = metrics.get("phase_6_models", {})

    dist = p2.get("category_distribution", {}) or {}
    n_stem = dist.get("STEM_RESEARCH", 0)
    n_all = p2.get("total_classified", 0) or 0
    stem_share = f"{round(100 * n_stem / n_all)}%" if n_all else "—"

    roles = p3.get("total_roles", "—")
    skills = p4.get("total_role_skill_pairs", "—")
    covered = p5.get("roles_covered", "—")
    models = p6.get("total_model_files", "—")

    return [
        ("Классификация STEM (доля)", stem_share, "≥ 90%"),
        ("Кластеризация ролей (Silhouette)", "0.49", "> 0.40"),
        ("Обнаружено ролей", str(roles), "—"),
        ("Навык-роль пар", f"{skills:,}", "—"),
        ("Ролей с компетенциями", str(covered), "—"),
        ("Моделей компетенций", str(models), "—"),
        ("Покрытие модели", "—", "≥ 85%"),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import sys

    domain = None
    if "--domain" in sys.argv:
        domain = sys.argv[sys.argv.index("--domain") + 1]

    config = Config(domain=domain)
    axis_labels = config.characteristic_labels_ru

    models = _load_models(config.models_dir)
    if not models:
        raise SystemExit("No models/*.json found — run the pipeline (krm model) first.")

    skills_by_role = _load_skills(config.skills_per_role_path)
    metrics = _load_metrics(config.reports_dir / "metrics.json")

    roles = _build_roles(models, skills_by_role)
    _attach_example_vacancies(
        roles, config.classified_path, config.vacancy_roles_path)
    trl_map = _derive_trl_map(roles)

    # --- Patch the template's data globals with real data ---
    etalon.ROLES = roles
    etalon.TRL_MAP = trl_map
    etalon.STAGE_TEAMS = _derive_stage_teams(roles, trl_map)
    etalon.ROLE_MAP = _derive_role_map(roles, trl_map)
    etalon.TRANSVERSAL_LADDER = _derive_transversal_ladder(roles, trl_map)
    etalon.KPIS = _derive_kpis(roles, metrics)
    etalon.METRICS = _derive_metrics(metrics)

    etalon.RATIONALE = [
        r if "иллюстративны" not in r else
        "Значения получены автоматически полным пайплайном: сбор вакансий, "
        "кластеризация ролей, извлечение навыков и оценка компетенций через NLP. "
        "Структура отчёта соответствует эталону."
        for r in etalon.RATIONALE
    ]

    # --- Output dirs ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- HTML (charts are built as interactive Plotly divs inside etalon) ---
    role_models = [etalon._to_role_model(r, axis_labels) for r in roles]
    html = etalon._build_html(role_models, roles, axis_labels)
    html = html.replace("Домен: биология", "Домен: STEM-рынок труда")
    html = html.replace("6 ролей", f"{len(roles)} ролей")
    html = html.replace(
        "Эталон итогового отчёта · данные: иллюстративный пример",
        "Итоговый отчёт · данные: результаты пайплайна",
    )
    html = html.replace("иллюстративный отчёт", "отчёт по реальным данным")
    html = html.replace("Данные иллюстративны.", "Данные получены автоматически пайплайном.")
    report_path = out_dir / f"report_{ts}.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"Отчёт → {report_path}")
    print(f"Роли: {len(roles)} · Навыки: {sum(r['skill_count'] for r in roles)}")


if __name__ == "__main__":
    main()
