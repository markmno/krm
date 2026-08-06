"""doit task definitions for KRM pipeline.

Each phase is a doit task with file dependencies on the Parquet output
of the previous phase. This enables incremental execution: doit skips
tasks whose output is newer than their input.

Usage:
    doit list          # List all tasks
    doit               # Run all (incremental)
    doit -n 4          # Run with 4 parallel workers
    doit clean         # Remove generated files
"""

from __future__ import annotations

from doit.task import clean_targets


DOIT_CONFIG = {
    "default_tasks": ["all"],
    "verbosity": 2,
    "num_process": 4,
}


# --- File paths ----------------------------------------------------------------


def _output(name: str) -> str:
    return f"data/{name}.parquet"


def _report(name: str) -> str:
    return f"reports/{name}"


# --- Phase 1: Collect ----------------------------------------------------------


def task_collect():
    """Phase 1: Scrape HH.ru vacancies into DuckDB."""
    return {
        "actions": ["python -m src.krm.cli collect"],
        "targets": ["data/krm.duckdb"],
        "uptodate": [False],  # Always check for new vacancies
        "clean": True,
        "doc": "Scrape HH.ru API for STEM research vacancies",
    }


# --- Phase 2: Classify ---------------------------------------------------------


def task_classify():
    """Phase 2: Classify vacancies as STEM/IT/non-STEM."""
    return {
        "actions": ["python -m src.krm.cli classify"],
        "targets": [_output("classified")],
        "file_dep": ["data/krm.duckdb"],
        "clean": True,
        "doc": "4-tier STEM/IT classifier using NLI",
    }


# --- Phase 3: Roles ------------------------------------------------------------


def task_roles():
    """Phase 3: Discover roles via embedding + UMAP + HDBSCAN clustering."""
    return {
        "actions": ["python -m src.krm.cli roles"],
        "targets": [_output("roles")],
        "file_dep": [_output("classified")],
        "clean": True,
        "doc": "HDBSCAN clustering of job title embeddings into roles",
    }


# --- Phase 4: Skills -----------------------------------------------------------


def task_skills():
    """Phase 4: Extract skills per role using ESCO taxonomy."""
    return {
        "actions": ["python -m src.krm.cli skills"],
        "targets": [_output("skills_per_role")],
        "file_dep": [_output("roles"), _output("classified")],
        "clean": True,
        "doc": "ESCO-based skill extraction with TF-IDF weighting",
    }


# --- Phase 5: Axes -------------------------------------------------------------


def task_axes():
    """Phase 5: Map skills to 6 competency axes via zero-shot NLI."""
    return {
        "actions": ["python -m src.krm.cli axes"],
        "targets": [_output("axis_scores")],
        "file_dep": [_output("skills_per_role")],
        "clean": True,
        "doc": "Zero-shot NLI mapping of skills to fixed competency axes",
    }


# --- Phase 6: Model ------------------------------------------------------------


def task_model():
    """Phase 6: Build competency models and spider charts."""
    return {
        "actions": ["python -m src.krm.cli model"],
        "targets": [_output("axis_scores")],  # Uses models/ and reports/ dirs
        "file_dep": [_output("axis_scores"), _output("roles")],
        "clean": [clean_targets],
        "doc": "Competency role models with spider charts (7-level scale)",
    }


# --- Phase 7: Validate ---------------------------------------------------------


def task_validate():
    """Phase 7: Validate pipeline and generate reports."""
    return {
        "actions": ["python -m src.krm.cli validate"],
        "targets": [_report("metrics.json"), _report("validation_report.md")],
        "file_dep": [_output("classified"), _output("roles"), _output("skills_per_role"), _output("axis_scores")],
        "clean": True,
        "doc": "End-to-end pipeline validation metrics and report",
    }


# --- Meta-tasks ----------------------------------------------------------------


def task_all():
    """Run all phases in order."""
    return {
        "actions": None,
        "task_dep": ["collect", "classify", "roles", "skills", "axes", "model", "validate"],
        "doc": "Run complete pipeline (all 7 phases)",
    }


def task_clean_all():
    """Remove all generated files."""
    return {
        "actions": [
            "rm -f data/krm.duckdb data/*.parquet",
            "rm -rf models/ reports/",
        ],
        "doc": "Remove all pipeline outputs",
    }
