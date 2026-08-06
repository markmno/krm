"""CLI entry point for KRM pipeline.

Usage:
    krm collect    # Phase 1: Scrape HH.ru vacancies
    krm classify   # Phase 2: Classify STEM/IT/non-STEM
    krm roles      # Phase 3: Discover roles via clustering
    krm skills     # Phase 4: Extract skills per role
    krm axes       # Phase 5: Map skills to competency axes
    krm model      # Phase 6: Build competency models + charts
    krm validate   # Phase 7: Validate and report
    krm all        # Run all phases sequentially
"""

from __future__ import annotations

import sys
from pathlib import Path

from krm.config import Config


def _get_config(config_path: str | None = None) -> Config:
    path = Path(config_path) if config_path else None
    cfg = Config(path)
    cfg.ensure_dirs()
    return cfg


def cmd_collect(config_path: str | None = None) -> None:
    """Phase 1: Scrape HH.ru vacancies."""
    from krm.phase_1_collect import collect

    cfg = _get_config(config_path)
    n = collect(cfg)
    print(f"Collected {n} new vacancies")


def cmd_classify(config_path: str | None = None) -> None:
    """Phase 2: Classify STEM/IT/non-STEM."""
    from krm.phase_2_classify import classify_vacancies

    cfg = _get_config(config_path)
    df = classify_vacancies(cfg)
    counts = df["stem_category"].value_counts().to_dict()
    print("Classification results:")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")


def cmd_roles(config_path: str | None = None) -> None:
    """Phase 3: Discover roles via HDBSCAN clustering."""
    from krm.phase_3_roles import discover_roles

    cfg = _get_config(config_path)
    df = discover_roles(cfg)
    n_roles = df[df["role_id"] != -1]["role_id"].nunique()
    n_noise = (df["role_id"] == -1).sum()
    print(f"Discovered {n_roles} roles ({n_noise} noise points)")
    for _, row in df[df["role_id"] != -1].iterrows():
        print(f"  {row['role_id']}: {row['role_label']} ({row['member_count']} vacancies)")


def cmd_skills(config_path: str | None = None) -> None:
    """Phase 4: Extract skills per role."""
    from krm.phase_4_skills import extract_skills

    cfg = _get_config(config_path)
    df = extract_skills(cfg)
    n_skills = df["skill_canonical_name"].nunique()
    n_roles = df["role_id"].nunique()
    print(f"Extracted {n_skills} unique skills across {n_roles} roles")


def cmd_axes(config_path: str | None = None) -> None:
    """Phase 5: Map skills to competency axes."""
    from krm.phase_5_axes import map_to_axes

    cfg = _get_config(config_path)
    df = map_to_axes(cfg)
    print(f"Axis scores computed for {df['role_id'].nunique()} roles")
    for axis_id in cfg.axis_ids:
        label = {a["id"]: a["label_ru"] for a in cfg.axes}[axis_id]
        mean_score = df[df["axis_id"] == axis_id]["proficiency"].mean()
        print(f"  {label}: mean={mean_score:.2f}")


def cmd_model(config_path: str | None = None) -> None:
    """Phase 6: Build competency models and spider charts."""
    from krm.phase_6_model import build_models

    cfg = _get_config(config_path)
    models = build_models(cfg)
    print(f"Built {len(models)} competency models")
    for m in models:
        print(f"  {m['role_id']}: {m['role_label']}")


def cmd_validate(config_path: str | None = None) -> None:
    """Phase 7: Validate pipeline and generate reports."""
    from krm.phase_7_validate import validate

    cfg = _get_config(config_path)
    report = validate(cfg)
    integration = report.get("phases", {}).get("integration", {})
    for k, v in integration.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
    for phase_name, phase_data in report.get("phases", {}).items():
        if phase_name == "integration":
            continue
        if isinstance(phase_data, dict) and "status" in phase_data:
            print(f"  {phase_name}: {phase_data['status']}")


def cmd_all(config_path: str | None = None) -> None:
    """Run all 7 phases sequentially."""
    steps = [
        ("Phase 1: Collect vacancies", cmd_collect),
        ("Phase 2: Classify STEM/IT", cmd_classify),
        ("Phase 3: Discover roles", cmd_roles),
        ("Phase 4: Extract skills", cmd_skills),
        ("Phase 5: Map to axes", cmd_axes),
        ("Phase 6: Build models", cmd_model),
        ("Phase 7: Validate", cmd_validate),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        try:
            fn(config_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: krm <command> [--config config.yaml]")
        print("Commands: collect, classify, roles, skills, axes, model, validate, all")
        sys.exit(1)

    command = sys.argv[1]
    config_path: str | None = None

    # Parse --config flag
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]

    commands: dict[str, object] = {
        "collect": cmd_collect,
        "classify": cmd_classify,
        "roles": cmd_roles,
        "skills": cmd_skills,
        "axes": cmd_axes,
        "model": cmd_model,
        "validate": cmd_validate,
        "all": cmd_all,
    }

    if command not in commands:
        print(f"Unknown command: {command}")
        sys.exit(1)

    fn = commands[command]
    if fn is not None and hasattr(fn, "__call__"):
        fn(config_path)  # type: ignore[operator]


if __name__ == "__main__":
    main()
