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


# --- Phase 2.5: Characteristics ------------------------------------------------


def task_phase25():
    """Phase 2.5: Score vacancy descriptions against competency hypotheses."""
    return {
        "actions": ["python -m src.krm.cli phase25"],
        "targets": [_output("characteristics")],
        "file_dep": [_output("classified")],
        "clean": True,
        "doc": "Vacancy-level characteristic scoring (7 hard axes + experience)",
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
        "targets": [_output("skills_per_role"), _output("vacancy_roles")],
        "file_dep": [_output("roles"), _output("classified")],
        "clean": True,
        "doc": "ESCO-based skill extraction with TF-IDF weighting",
    }


# --- Phase 5: Characteristics -------------------------------------------------


def task_characteristics():
    """Phase 5: Map skills to 7 competency characteristics via zero-shot NLI."""
    return {
        "actions": ["python -m src.krm.cli characteristics"],
        "targets": [_output("characteristic_scores"), _output("skill_characteristic_scores")],
        "file_dep": [_output("skills_per_role")],
        "clean": True,
        "doc": "Zero-shot NLI mapping of skills to fixed competency characteristics",
    }


# --- Phase 5b: Soft competences ------------------------------------------------


def task_soft():
    """Phase 5b: Archetype classification + soft-competence scoring."""
    return {
        "actions": ["python -m src.krm.cli soft"],
        "targets": [_output("soft_scores"), _output("role_archetypes")],
        "file_dep": [_output("skills_per_role"), _output("vacancy_roles"), _output("classified")],
        "clean": True,
        "doc": "Per-role archetype classification and 4-axis soft-competence scoring",
    }


# --- Phase 6: Model ------------------------------------------------------------


def task_model():
    """Phase 6: Build competency models and spider charts."""
    return {
        "actions": ["python -m src.krm.cli model"],
        "targets": [_output("characteristic_scores")],  # Uses models/ and reports/ dirs
        "file_dep": [
            _output("characteristic_scores"),
            _output("roles"),
            _output("soft_scores"),
            _output("role_archetypes"),
            _output("vacancy_roles"),
            _output("skill_characteristic_scores"),
            _output("skills_per_role"),
            _output("characteristics"),
        ],
        "clean": [clean_targets],
        "doc": "Competency role models with spider charts (7-level scale)",
    }


# --- Phase 7: Validate ---------------------------------------------------------


def task_validate():
    """Phase 7: Validate pipeline and generate reports."""
    return {
        "actions": ["python -m src.krm.cli validate"],
        "targets": [_report("metrics.json"), _report("validation_report.md")],
        "file_dep": [_output("classified"), _output("roles"), _output("skills_per_role"), _output("characteristic_scores")],
        "clean": True,
        "doc": "End-to-end pipeline validation metrics and report",
    }


# --- Meta-tasks ----------------------------------------------------------------


def task_all():
    """Run all phases in order."""
    return {
        "actions": None,
        "task_dep": [
            "collect",
            "classify",
            "phase25",
            "roles",
            "skills",
            "characteristics",
            "soft",
            "model",
            "validate",
        ],
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


# --- Phase 0: Census ------------------------------------------------------------


def task_phase0_census():
    """Phase 0: Wayback CDX coverage census across all sources."""
    return {
        "actions": ["python -m krm.cli phase0-census"],
        "targets": ["data/phase0/census/census_report.json"],
        "uptodate": [False],
        "doc": "CDX API coverage estimates for hh.ru and LinkedIn Wayback",
        "clean": True,
    }


# --- Phase 0: Historical Data Collectors ----------------------------------------


_PHASE0_DB = "data/phase0/historical.duckdb"


def task_phase0_hhru():
    """Phase 0: Collect hh.ru historical vacancies from Wayback Machine."""
    return {
        "actions": ["python -m krm.cli phase0-hhru"],
        "doc": "hh.ru Wayback Machine CDX → fetch → parse → store",
    }


def task_phase0_linkedin():
    """Phase 0: Collect LinkedIn historical job postings from Wayback Machine."""
    return {
        "actions": ["python -m krm.cli phase0-linkedin"],
        "doc": "LinkedIn Wayback Machine CDX → fetch → parse → store",
    }


def task_phase0_trudvsem():
    """Phase 0: Collect Trudvsem open data vacancies (hh.ru sourced)."""
    return {
        "actions": ["python -m krm.cli phase0-trudvsem"],
        "doc": "Trudvsem.ru open data API collector",
    }


def task_phase0_rostud():
    """Phase 0: Collect Rostrud bulk CSV/XLSX vacancy data."""
    return {
        "actions": ["python -m krm.cli phase0-rostud"],
        "doc": "Rostrud dataset directory scanner (CSV/XLSX)",
    }


# --- Phase 0: Meta-tasks --------------------------------------------------------


def task_phase0_all():
    """Phase 0: Run census + all historical data collectors."""
    return {
        "actions": None,
        "task_dep": [
            "phase0_census",
            "phase0_hhru",
            "phase0_linkedin",
            "phase0_trudvsem",
            "phase0_rostud",
        ],
        "doc": "Run complete Phase 0 pipeline",
    }


def task_phase0_verify():
    """Phase 0: Print collection run summaries from the Phase 0 database."""
    return {
        "actions": ["python -m krm.cli phase0-verify"],
        "doc": "Verify Phase 0 collection run summaries",
    }


def task_phase0_merge():
    """Phase 0: Merge Phase 0 historical database into main pipeline database."""
    return {
        "actions": ["python -m krm.cli phase0-merge"],
        "file_dep": [_PHASE0_DB],
        "targets": ["data/krm.duckdb"],
        "doc": "Merge Phase 0 raw_vacancies into the main pipeline DB",
    }
