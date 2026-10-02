"""Prefect flows wrapping the existing KRM pipeline phases.

Each phase of ``krm.cli.cmd_all`` becomes a thin ``@task`` wrapper that calls
the existing phase function unchanged (the task body only imports the phase
module lazily and forwards ``Config`` — mirroring how ``krm.cli`` does it).

``pipeline_flow`` runs the phases in the same order as ``cmd_all`` and adds one
*real* runtime branch (a dynamic DAG, decided inside the flow body at run time):
after classification it checks whether any STEM vacancies were found and either

    * runs the **LLM-backed** skill-extraction phase (Phase 2b), gated on a
      healthy llama-server (``wait_for_llama``), or
    * runs the **deterministic** spaCy noun-phrase path from the same
      ``phase_2b_extract_skills`` module.

Both branches call ``phase_2b_extract_skills.extract_skills(config)``; the
deterministic branch forces ``use_llm_phase_2b`` off so the module's own
``_extract_skills_noun_phrases`` fallback is used.

Adding a new phase later = add one ``@task`` + one call in ``pipeline_flow``.

Run:

    python -m flows.pipeline              # one-shot full pipeline
    python -m flows.pipeline --serve      # serve collect_flow on a daily cron
    python -m flows.pipeline --serve-pipeline  # serve pipeline_flow on an interval

Schedules use the verified Prefect 3.8 API: ``from prefect.schedules import
Cron, Interval`` (NOT ``prefect.client.schemas.schedules``), and ``flow.serve``
accepts ``cron`` / ``interval`` / ``rrule`` / ``schedule`` / ``schedules``
directly.  The string shorthand ``cron="0 3 * * *"`` equals ``Cron("0 3 * * *")``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from prefect import flow, task
from prefect.deployments import run_deployment
from prefect.schedules import Cron, Interval

from krm.config import Config

_ROOT = Path(__file__).parent.parent

# STEM_RESEARCH is the unambiguous "STEM vacancy" category produced by
# Phase 2 (krm.phase_2_classify).  It is imported lazily inside the classify
# task, next to the phase function it labels.


# ---------------------------------------------------------------------------
# Config helpers (mirror krm.cli._get_config)
# ---------------------------------------------------------------------------


def _get_config(config_path: str | None = None) -> Config:
    """Build a ``Config`` and ensure output dirs exist (mirrors ``krm.cli``)."""
    path = Path(config_path) if config_path else None
    cfg = Config(path)
    cfg.ensure_dirs()
    return cfg


def _force_deterministic(config: Config) -> Config:
    """Return a copy of ``config`` with the LLM phase-2b flag forced off.

    The LLM and deterministic skill-extraction paths share the single
    ``phase_2b_extract_skills.extract_skills`` entrypoint; it dispatches on
    ``config.use_llm_phase_2b``.  Forcing the flag off routes the deterministic
    branch through the spaCy noun-phrase fallback that lives in that module.
    """
    import copy

    cfg = copy.deepcopy(config)
    phases = cfg._data.setdefault("llm", {}).setdefault("phases", {})
    phases.setdefault("phase_2b_skills", {})["use_llm"] = False
    return cfg


# ---------------------------------------------------------------------------
# Phase tasks (thin wrappers around the existing phase functions)
# ---------------------------------------------------------------------------


@task(retries=3, retry_delay_seconds=30, tags=["krm", "collect", "phase-1"])
def collect_task(config: Config) -> int:
    """Phase 1: scrape hh.ru vacancies via the public website (no API key)."""
    from krm.phase_1_site import collect_site

    return collect_site(config)


@task(retries=1, retry_delay_seconds=10, tags=["krm", "classify", "phase-2"])
def classify_task(config: Config) -> int:
    """Phase 2: classify vacancies; return the number of STEM vacancies found."""
    from krm.phase_2_classify import STEM_RESEARCH, classify_vacancies

    classified = classify_vacancies(config)
    if classified is None or classified.empty:
        return 0
    return int((classified["stem_category"] == STEM_RESEARCH).sum())


@task(retries=1, retry_delay_seconds=10, tags=["krm", "skills", "phase-2b", "llm"])
def extract_skills_llm_task(config: Config) -> int:
    """Phase 2b (LLM): extract Russian skill phrases via the shared LLM client.

    Requires ``config.use_llm_phase_2b`` to be truthy (the default in
    ``config.yaml``).  The caller gates this task behind ``wait_for_llama``.
    """
    from krm.phase_2b_extract_skills import extract_skills

    _, vocab_df = extract_skills(config)
    return int(len(vocab_df))


@task(retries=1, retry_delay_seconds=10, tags=["krm", "skills", "phase-2b", "deterministic"])
def extract_skills_deterministic_task(config: Config) -> int:
    """Phase 2b (deterministic): spaCy noun-phrase fallback from the same module."""
    from krm.phase_2b_extract_skills import extract_skills

    _, vocab_df = extract_skills(_force_deterministic(config))
    return int(len(vocab_df))


@task(retries=1, retry_delay_seconds=10, tags=["krm", "characteristics", "phase-2.5"])
def phase25_task(config: Config) -> int:
    """Phase 2.5: extract vacancy-level characteristics for downstream grounding."""
    from krm.phase_2_5_characteristics import run_pipeline

    df, _title_chars = run_pipeline(config)
    return int(len(df))


@task(retries=1, retry_delay_seconds=10, tags=["krm", "roles", "phase-3"])
def roles_task(config: Config) -> int:
    """Phase 3: discover roles via HDBSCAN clustering."""
    from krm.phase_3_roles import discover_roles

    df = discover_roles(config)
    if df is None or df.empty:
        return 0
    return int(df[df["role_id"] != -1]["role_id"].nunique())


@task(retries=1, retry_delay_seconds=10, tags=["krm", "skills", "phase-4"])
def skills_task(config: Config) -> int:
    """Phase 4: extract skills per role."""
    from krm.phase_4_skills import extract_skills

    df = extract_skills(config)
    return int(df["skill_canonical_name"].nunique())


@task(retries=1, retry_delay_seconds=10, tags=["krm", "characteristics", "phase-5"])
def characteristics_task(config: Config) -> int:
    """Phase 5: map skills to competency characteristics."""
    from krm.phase_5_axes import map_to_characteristics

    df = map_to_characteristics(config)
    if df.empty:
        return 0
    return int(df["role_id"].nunique())


@task(retries=1, retry_delay_seconds=10, tags=["krm", "soft", "phase-5b"])
def soft_task(config: Config) -> int:
    """Phase 5b: classify role archetypes and score soft competences."""
    from krm.phase_5b_soft import classify_role_archetypes, score_soft_competences

    archetypes = classify_role_archetypes(config)
    score_soft_competences(config)
    if archetypes.empty:
        return 0
    return int(archetypes["role_id"].nunique())


@task(retries=1, retry_delay_seconds=10, tags=["krm", "model", "phase-6"])
def model_task(config: Config) -> int:
    """Phase 6: build competency models and spider charts."""
    from krm.phase_6_model import build_models

    models = build_models(config)
    return int(len(models))


@task(retries=1, retry_delay_seconds=10, tags=["krm", "validate", "phase-7"])
def validate_task(config: Config) -> dict[str, Any]:
    """Phase 7: validate the pipeline and return the report summary."""
    from krm.phase_7_validate import validate

    return validate(config)


@task(retries=1, retry_delay_seconds=10, tags=["krm", "report", "phase-8"])
def report_task(config: Config) -> None:
    """Phase 8: generate the self-contained HTML report from pipeline data."""
    cmd = [sys.executable, "-m", "scripts.generate_report"]
    subprocess.run(cmd, cwd=_ROOT, check=True)


@task(retries=0, tags=["krm", "llm", "gate"])
def wait_for_llama(
    config: Config,
    attempts: int = 10,
    delay_seconds: float = 5.0,
) -> None:
    """Gate the LLM branch: poll the llama-server ``/health`` endpoint.

    ``config.llm_base_url`` is an OpenAI-compatible base URL (e.g.
    ``http://localhost:8080/v1``); the llama.cpp health check lives at the
    server root (``/health``).  Raises ``RuntimeError`` after ``attempts``
    failed polls so the branch fails fast instead of hanging the pipeline.
    """
    root = config.llm_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    health_url = f"{root}/health"

    for attempt in range(1, attempts + 1):
        try:
            resp = httpx.get(health_url, timeout=10.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if attempt < attempts:
            time.sleep(delay_seconds)

    msg = f"llama-server not healthy at {health_url} after {attempts} attempts"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


@flow(name="krm-collect", description="Scheduled hh.ru vacancy scraping (Phase 1).")
def collect_flow(config_path: str | None = None) -> int:
    """Scrape hh.ru vacancies into DuckDB.  Meant to run on a cron schedule."""
    config = _get_config(config_path)
    return collect_task(config)


@flow(
    name="krm-pipeline",
    description="Full KRM pipeline with a dynamic STEM/LLM branch.",
    retries=1,
    retry_delay_seconds=60,
)
def pipeline_flow(
    config_path: str | None = None,
    run_collect: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline in ``krm.cli.cmd_all`` order, with a dynamic branch.

    Phase order (identical to ``cmd_all``):

        collect → classify → phase2.5 → roles → skills → characteristics
        → soft → model → validate → report

    The *dynamic* branch (a real Python ``if/else`` evaluated at run time, so
    Prefect records two different DAGs): after classification, if any STEM
    vacancies were found we gate on a healthy llama-server and run the
    LLM-backed skill-extraction phase (Phase 2b); otherwise we run the
    deterministic spaCy noun-phrase path from the same module.
    """
    config = _get_config(config_path)
    summary: dict[str, Any] = {}

    if run_collect:
        summary["collected"] = collect_task(config)
    else:
        summary["collected"] = "skipped"

    n_stem = classify_task(config)
    summary["stem_vacancies"] = n_stem

    # -- Dynamic branch: LLM vs deterministic skill extraction (Phase 2b) -----
    if n_stem > 0:
        wait_for_llama(config)
        summary["skills_2b"] = extract_skills_llm_task(config)
    else:
        summary["skills_2b"] = extract_skills_deterministic_task(config)

    summary["phase25_scores"] = phase25_task(config)
    summary["roles"] = roles_task(config)
    summary["skills"] = skills_task(config)
    summary["characteristic_scores"] = characteristics_task(config)
    summary["soft_scores"] = soft_task(config)
    summary["models"] = model_task(config)
    summary["validation"] = validate_task(config)
    report_task(config)
    summary["report"] = "done"

    return summary


# ---------------------------------------------------------------------------
# Downstream triggering (verified Prefect 3.8 entrypoint)
# ---------------------------------------------------------------------------


def trigger_pipeline_deployment(
    deployment: str = "krm-pipeline/default",
    **parameters: Any,
) -> Any:
    """Trigger an already-deployed ``pipeline_flow`` run by name.

    Uses the verified ``prefect.deployments.run_deployment`` entrypoint to
    submit a flow run against an existing deployment — no Automation /
    TriggerType event API needed.  Useful for chaining (e.g. a collect run can
    kick off the pipeline):

        trigger_pipeline_deployment("krm-pipeline/default")
    """
    return run_deployment(deployment, parameters=parameters or None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--serve" in sys.argv:
        # Serve the collect flow on a daily cron (03:00 local time).  The
        # string shorthand ``cron="0 3 * * *"`` is equivalent to the explicit
        # ``Cron("0 3 * * *")`` schedule object used here.
        collect_flow.serve(
            name="krm-collect",
            schedules=[Cron("0 3 * * *")],
            tags=["krm", "collect"],
        )
    elif "--serve-pipeline" in sys.argv:
        # Serve the pipeline itself on an explicit 6-hour Interval schedule
        # (``interval=21600`` is the equivalent shorthand).
        pipeline_flow.serve(
            name="krm-pipeline",
            schedules=[Interval(6 * 60 * 60)],
            tags=["krm", "pipeline"],
        )
    else:
        # One-shot full pipeline run.
        pipeline_flow()
