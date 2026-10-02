# Prefect orchestration for KRM

Additive Prefect flows/tasks that wrap the existing `krm` phase functions
without touching `krm.cli` or `dodo.py`.

## Flows

| Flow | Purpose | Schedule |
|---|---|---|
| `collect_flow` | Phase 1: scrape hh.ru vacancies into DuckDB (`krm.phase_1_site.collect_site`) | daily cron (03:00) |
| `pipeline_flow` | Full pipeline in `cmd_all` order, with one dynamic branch | 6-hour interval |

Each phase is a thin `@task` that imports the phase module lazily and forwards
the `Config`, exactly as `krm.cli` does. Adding a phase = add one `@task` + one
call in `pipeline_flow`.

## Phase order (faithful to `krm.cli.cmd_all`)

```
collect → classify → phase2.5 → roles → skills → characteristics
→ soft → model → validate → report
```

## The dynamic branch

Inside `pipeline_flow`, after `classify`:

```python
n_stem = classify_task(config)

if n_stem > 0:
    wait_for_llama(config)                       # poll llama-server /health
    extract_skills_llm_task(config)              # LLM skill extraction (Phase 2b)
else:
    extract_skills_deterministic_task(config)    # spaCy noun-phrase fallback
```

This is a real Python `if/else` evaluated at run time, so Prefect records two
different DAGs. Both branches call `phase_2b_extract_skills.extract_skills`; the
deterministic branch forces `use_llm_phase_2b` off so the module's own
`_extract_skills_noun_phrases` fallback runs.

## Running

```bash
# One-shot full pipeline run
.venv/bin/python -m flows.pipeline

# Serve collect_flow on a daily cron (03:00)
.venv/bin/python -m flows.pipeline --serve

# Serve pipeline_flow on a 6-hour interval
.venv/bin/python -m flows.pipeline --serve-pipeline
```

Scheduling uses the verified Prefect 3.8 API: `from prefect.schedules import
Cron, Interval` (not `prefect.client.schemas.schedules`) and `flow.serve`, which
accepts `cron` / `interval` / `rrule` / `schedule` / `schedules` directly. The
string shorthand `cron="0 3 * * *"` equals `Cron("0 3 * * *")`; `interval=21600`
equals `Interval(21600)`.

## Triggering the pipeline from another deployment

`trigger_pipeline_deployment("krm-pipeline/default")` uses the verified
`prefect.deployments.run_deployment` entrypoint to submit a pipeline run by
name — e.g. from a `collect_flow` completion hook.
