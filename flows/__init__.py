"""Prefect orchestration for the KRM pipeline.

Wraps the existing ``krm`` phase functions (``krm.cli`` entrypoints) as
Prefect flows/tasks, so the pipeline can be scheduled and observed without
touching ``krm.cli`` or ``dodo.py``.  Everything here is additive.

Exposed flows:

    - ``collect_flow``  — scheduled live scraping (Phase 1, ``krm.phase_1_site``).
    - ``pipeline_flow`` — classify → dynamic STEM/LLM branch → cluster → report.
"""

from flows.pipeline import collect_flow, pipeline_flow

__all__ = ["collect_flow", "pipeline_flow"]
