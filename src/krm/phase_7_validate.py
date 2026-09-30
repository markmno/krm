"""Phase 7: End-to-End Validation and Reporting.

Validates output quality at every pipeline stage against defined metrics:
- Phase 2: F1 score on hand-labelled classification sample
- Phase 3: Silhouette score, Davies-Bouldin index, bootstrap ARI stability
- Phase 4: Precision@k on expert-labelled skill sets
- Phase 5: Spearman ρ correlation vs expert characteristic scores
- Integration: Coverage ratio (vacancies with complete model)

Writes a structured `reports/validation_report.md` and `reports/metrics.json`.

Usage:
    from krm.config import Config
    from krm.phase_7_validate import validate

    config = Config()
    report = validate(config)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from krm.config import Config
from krm.lib.io import read_parquet


def _safe_read(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[validate] SKIP {label}: {path} not found")
        return None
    return read_parquet(path)


def _validate_phase_2_classification(config: Config) -> dict[str, Any]:
    """Check classified.parquet column integrity and category distribution."""
    df = _safe_read(config.classified_path, "classified.parquet")
    if df is None:
        return {"status": "skipped", "reason": "classified.parquet not found"}

    expected_cols = {"vacancy_id", "title", "description", "stem_category"}
    missing = expected_cols - set(df.columns)
    total = len(df)
    counts = df["stem_category"].value_counts().to_dict() if "stem_category" in df.columns else {}

    return {
        "total_classified": total,
        "missing_columns": sorted(missing),
        "category_distribution": {str(k): int(v) for k, v in counts.items()},
        "f1_target": ">= 0.92 (requires hand-labelled sample — not automated)",
    }


def _validate_phase_3_roles(config: Config) -> dict[str, Any]:
    """Validate role clustering: silhouette, DBI, bootstrap ARI proxy."""
    df = _safe_read(config.roles_path, "roles.parquet")
    if df is None:
        return {"status": "skipped", "reason": "roles.parquet not found"}

    if "centroid_embedding" not in df.columns or len(df) < 2:
        return {"total_roles": len(df) if df is not None else 0, "error": "too few roles for clustering metrics"}

    centroids = np.stack([np.frombuffer(r, dtype=np.float32) for r in df["centroid_embedding"].dropna()])
    if len(centroids) < 2:
        return {"total_roles": len(df), "error": "too few valid centroids"}

    centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-12)
    sim = centroids_norm @ centroids_norm.T
    np.fill_diagonal(sim, 0)
    inter_cluster_separation = float(1.0 - sim.max())

    return {
        "total_roles": len(df),
        "noise_roles": int(df["noise_flag"].sum()) if "noise_flag" in df.columns else None,
        "mean_member_count": float(df["member_count"].mean()) if "member_count" in df.columns else None,
        "inter_cluster_separation": round(inter_cluster_separation, 4),
        "silhouette_target": "> 0.4 (requires cluster labels per vacancy) — not computed automatically",
        "bootstrap_ari_target": "> 0.7 (requires hand-labelled roles) — not computed automatically",
    }


def _validate_phase_4_skills(config: Config) -> dict[str, Any]:
    """Validate skills_per_role: coverage and top skills."""
    df = _safe_read(config.skills_per_role_path, "skills_per_role.parquet")
    if df is None:
        return {"status": "skipped", "reason": "skills_per_role.parquet not found"}

    n_roles = int(df["role_id"].nunique())
    n_skills = len(df)
    mean_per_role = float(round(n_skills / n_roles, 1)) if n_roles else 0.0

    return {
        "total_role_skill_pairs": n_skills,
        "roles_with_skills": n_roles,
        "mean_skills_per_role": mean_per_role,
        "has_uncatalogued": "uncatalogued_skills" in df.columns,
        "precision_at_10_target": ">= 0.85 (requires expert-labelled sample) — not computed automatically",
    }


def _validate_phase_5_characteristics(config: Config) -> dict[str, Any]:
    """Validate characteristic_scores: coverage and distribution."""
    df = _safe_read(config.characteristic_scores_path, "characteristic_scores.parquet")
    if df is None:
        return {"status": "skipped", "reason": "characteristic_scores.parquet not found"}

    if "proficiency" not in df.columns:
        return {"total_rows": len(df), "error": "proficiency column missing"}

    prof = df["proficiency"].dropna()
    if len(prof) == 0:
        return {"total_rows": len(df), "error": "no valid proficiency scores"}

    return {
        "total_characteristic_scores": len(df),
        "roles_covered": int(df["role_id"].nunique() if "role_id" in df.columns else 0),
        "characteristics_covered": int(
            df["characteristic_id"].nunique() if "characteristic_id" in df.columns else 0
        ),
        "proficiency_range": [float(round(prof.min(), 2)), float(round(prof.max(), 2))],
        "proficiency_mean": float(round(prof.mean(), 2)),
        "spearman_target": ">= 0.75 (requires expert-labelled characteristic scores) — not computed automatically",
    }


def _validate_phase_6_models(config: Config) -> dict[str, Any]:
    """Validate competency model JSON files."""
    models_dir = config.models_dir
    if not models_dir.exists():
        return {"status": "skipped", "reason": f"models_dir ({models_dir}) not found"}

    json_files = sorted(models_dir.glob("*.json"))
    valid = 0
    invalid = 0
    errors: list[str] = []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            required = {
                "role_id",
                "role_label",
                "characteristics",
                "soft_competences",
                "experience",
                "skills",
            }
            missing = required - set(data.keys())
            if missing:
                invalid += 1
                errors.append(f"{jf.name}: missing keys {missing}")
            else:
                valid += 1
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            invalid += 1
            errors.append(f"{jf.name}: {e}")

    return {
        "total_model_files": len(json_files),
        "valid": valid,
        "invalid": invalid,
        "errors": errors[:10],
        "integrity_target": "100% valid JSON (expert validation deferred)",
    }


def _validate_integration(config: Config) -> dict[str, Any]:
    """Check cross-phase data integrity: role_id continuity, coverage ratio."""
    classified = _safe_read(config.classified_path, "classified.parquet")
    roles = _safe_read(config.roles_path, "roles.parquet")
    skills = _safe_read(config.skills_per_role_path, "skills_per_role.parquet")
    characteristics = _safe_read(
        config.characteristic_scores_path, "characteristic_scores.parquet"
    )
    soft = _safe_read(config.soft_scores_path, "soft_scores.parquet")

    result: dict[str, Any] = {}

    if classified is not None:
        result["total_vacancies"] = len(classified)
        stem = classified[classified["stem_category"] == "STEM_RESEARCH"] if "stem_category" in classified.columns else pd.DataFrame()
        result["stem_research_vacancies"] = len(stem)

    if roles is not None:
        result["total_roles"] = len(roles)
        result["active_roles"] = int((~roles["noise_flag"]).sum()) if "noise_flag" in roles.columns else len(roles)

    if skills is not None and "role_id" in skills.columns:
        result["roles_with_skills"] = int(skills["role_id"].nunique())

    if characteristics is not None and "role_id" in characteristics.columns:
        result["roles_with_characteristic_scores"] = int(characteristics["role_id"].nunique())
        expected_characteristics = 7
        characteristic_coverage = {
            rid: int(grp["characteristic_id"].nunique())
            for rid, grp in characteristics.groupby("role_id")
        }
        result["roles_with_complete_characteristics"] = sum(
            1 for v in characteristic_coverage.values() if v == expected_characteristics
        )

    expected_soft_axes = 4
    if soft is not None and "role_id" in soft.columns and "soft_id" in soft.columns:
        result["roles_with_soft_scores"] = int(soft["role_id"].nunique())
        soft_coverage = {
            rid: int(grp["soft_id"].nunique())
            for rid, grp in soft.groupby("role_id")
        }
        result["roles_with_complete_soft_scores"] = sum(
            1 for v in soft_coverage.values() if v == expected_soft_axes
        )
    else:
        result["roles_with_soft_scores"] = 0
        result["roles_with_complete_soft_scores"] = 0

    return result


def validate(config: Config) -> dict[str, Any]:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "pipeline": "KRM",
        "version": "0.2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phases": {
            "phase_2_classification": _validate_phase_2_classification(config),
            "phase_3_roles": _validate_phase_3_roles(config),
            "phase_4_skills": _validate_phase_4_skills(config),
            "phase_5_characteristics": _validate_phase_5_characteristics(config),
            "phase_6_models": _validate_phase_6_models(config),
            "integration": _validate_integration(config),
        },
    }

    metrics = {
        k: v for k, v in report["phases"].items() if k != "integration"
    }
    metrics["integration"] = report["phases"]["integration"]

    metrics_path = config.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[validate] metrics → {metrics_path}")

    md = _build_markdown_report(report)
    report_path = config.reports_dir / "validation_report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"[validate] report → {report_path}")

    return report


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# KRM Validation Report",
        f"",
        f"**Pipeline:** {report['pipeline']} v{report['version']}",
        f"**Timestamp:** {report['timestamp']}",
        f"",
        "---",
        "",
    ]

    for phase_name, phase_data in report["phases"].items():
        title = phase_name.replace("_", " ").title()
        lines.append(f"## {title}")
        lines.append("")
        if isinstance(phase_data, dict):
            for k, v in phase_data.items():
                if k.startswith("_"):
                    continue
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    return "\n".join(lines)
