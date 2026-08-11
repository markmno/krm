# HH-Competency: Implementation Plan

## Overview
**Goal:** Build a CLI tool that scrapes HH.ru, performs Russian NLP to extract skills from vacancies, clusters them into competency-role models for natural science specialties, and generates charts/reports.

**Tech Stack:** Python >=3.11, uv, Typer+Rich, DuckDB, httpx+asyncio, natasha+pymorphy3, msgspec, matplotlib+plotly, pandas, pytest

**Total:** 11 waves, 31 tasks, 11 atomic commits.

## User Requirements
1. Scrape HH.ru vacancies based on configurable specialty keywords (physics, biology, chemistry, ecology, biophysics)
2. Russian NLP: lemmatization (pymorphy3) + keyword extraction + stopword filtering
3. Skill clustering → competency-role models (like IT has Backend Engineer, ML Engineer)
4. Spider/radar charts showing skill clusters per role
5. Cross-specialty comparison (unique skills, shared skills, Jaccard overlap)
6. Mixed specialties (e.g., physics+biology → biophysics)
7. Output: terminal tables, CSV, Excel, HTML reports, charts
8. Phase separation: scrape ≠ analyze (re-analyze without re-scraping)

## Architecture
```
cli.py (Typer entry point)
├── commands/scrape.py  → HHClient + ScrapeEngine → DuckDB (raw_vacancies)
├── commands/analyze.py → NLPPipeline + SkillAnalyzer/SkillComparator/SkillClusterer
├── commands/stats.py   → Statistical tests (silhouette, χ², ANOVA, bootstrap)
├── commands/simulate.py → Monte Carlo simulation + curriculum optimization
├── commands/report.py  → viz/ (radar, bars, heatmap, salary charts)
├── stats/              → Hypothesis testing, stability, gap analysis
├── simulation/         → Monte Carlo, greedy optimization, what-if scenarios
└── methodology/        → Role model builder, profstandart mapping, FGOS comparison
```

## Waves

### W1: Project Bootstrap
**Commit:** `chore: bootstrap project skeleton with uv/pyproject.toml`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T1: Skeleton | pyproject.toml, .gitignore, all __init__.py, specialties.yml, data/.gitkeep | quick | programming | — |

### W2: Foundation
**Commit:** `feat: add data models, config, stopwords, and test infrastructure`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T2.1: Data Models | src/hh_competency/storage/models.py | quick | programming | T2.2, T2.3 |
| T2.2: Stopwords | src/hh_competency/nlp/stopwords.py | quick | programming | T2.1, T2.3 |
| T2.3: Test Infra | conftest.py, tests/conftest.py, tests/factories.py | quick | programming | T2.1, T2.2 |
| T2.4: Config | src/hh_competency/config.py | quick | programming | after T2.1 |

### W3: Services
**Commit:** `feat: add DuckDB storage, NLP pipeline, and HH.ru API client`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T3.1: DuckDB Storage | src/hh_competency/storage/db.py | deep | programming | T3.2, T3.3 |
| T3.2: NLP Pipeline | src/hh_competency/nlp/lemmatizer.py, pipeline.py | deep | programming | T3.1, T3.3 |
| T3.3: HH.ru API Client | src/hh_competency/api/client.py, rate_limiter.py | deep | programming | T3.1, T3.2 |

### W4: Scrape (parallel with W5)
**Commit:** `feat: add scrape engine and CLI commands`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T4.1: Scrape Engine | src/hh_competency/commands/scrape.py (engine) | deep | programming | — |
| T4.2: Scrape CLI | src/hh_competency/commands/scrape.py (Typer) | quick | programming, domain-cli | after T4.1 |

### W5: Analysis Engine (parallel with W4)
**Commit:** `feat: add analysis engine with skill frequencies, comparison, clustering`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T5.1: Skill Freqs | src/hh_competency/analysis/skills.py | deep | programming | T4.* |
| T5.2: Comparison | src/hh_competency/analysis/compare.py | deep | programming | after T5.1 |
| T5.3: Clustering | src/hh_competency/analysis/clustering.py | deep | programming | after T5.1 |

### W5B: Statistical Validation (runs after W5)
**Commit:** `feat: add statistical hypothesis testing for competency-role models`

**Depends on:** W5.1 (skill frequencies), W5.2 (comparison), W5.3 (clustering)
**Needed libraries:** scipy>=1.14.0, scikit-learn>=1.5.0, statsmodels>=0.14.0, numpy>=2.0.0

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T5B.1: Cluster Sig Tests | src/hh_competency/stats/clustering.py | deep | programming | T5B.2, T5B.3 |
| T5B.2: Specialty Diff Tests | src/hh_competency/stats/differentiation.py | deep | programming | T5B.1, T5B.3 |
| T5B.3: Stability + Gap Tests | src/hh_competency/stats/stability.py, gap.py | deep | programming | T5B.1, T5B.2 |

**T5B.1 — Cluster Significance Tests** (`src/hh_competency/stats/clustering.py`):
- Silhouette score with bootstrapped 95% CI (n_bootstrap=1000)
- Optimal k detection via silhouette elbow + gap statistic
- H₁ test: "Skills cluster into distinct role profiles" vs H₀ uniform
- Chi-square test on skill×specialty contingency table with Cramér's V effect size

**T5B.2 — Specialty Differentiation Tests** (`src/hh_competency/stats/differentiation.py`):
- χ² independence test: specialty vs skill distribution
- Post-hoc: which specialty pairs differ most (standardized residuals)
- ANOVA: skill diversity by experience level + Tukey HSD pairwise
- Effect size reporting: Cramér's V for χ², η² for ANOVA

**T5B.3 — Stability + Gap Tests** (`src/hh_competency/stats/stability.py`, `gap.py`):
- Bootstrap taxonomy stability (Rand index across 100 subsamples)
- Bootstrap Rand index 95% CI lower bound > 0.8 → stable
- Curriculum-market gap: Jaccard overlap FGOS vs market skills
- Level stratification: Kruskal-Wallis on skill counts by experience

**Test requirements:**
- `test_clustering.py`: bootstrap CI contains observed value, synthetic data with known clusters returns silhouette > 0.5
- `test_differentiation.py`: two clearly different skill distributions return p < 0.001
- `test_stability.py`: stable clusters (same data twice) return Rand index > 0.95

### W5C: Simulation & Optimization (runs after W5B)
**Commit:** `feat: add Monte Carlo simulation and curriculum optimization`

**Depends on:** W5B (statistical validation must pass before simulation)
**Needed libraries:** numpy (already added), scipy (already added)

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T5C.1: Monte Carlo Engine | src/hh_competency/simulation/monte_carlo.py | deep | programming | T5C.2 |
| T5C.2: Curriculum Optimizer | src/hh_competency/simulation/optimization.py, scenarios.py | deep | programming | T5C.1 |

**T5C.1 — Monte Carlo Engine** (`src/hh_competency/simulation/monte_carlo.py`):
- Vacancy-curriculum matching function: Jaccard similarity on skill sets
- Bootstrap resample vacancies (n=1000), compute mean coverage
- 95% CI on market coverage for any curriculum
- Distribution of individual vacancy match scores

**T5C.2 — Curriculum Optimizer** (`src/hh_competency/simulation/optimization.py`, `scenarios.py`):
- Greedy marginal gain optimization: add skill with highest coverage improvement
- What-if scenarios: FGOS-only, Market top-K, Optimized, Hybrid, Specialty-specific
- Scenario comparison: DataFrame output with mean coverage + CI per scenario
- Constraint: max_skills=50 parameter

**Test requirements:**
- `test_monte_carlo.py`: known curriculum on known data returns expected coverage
- `test_optimization.py`: greedy optimizer never decreases coverage
- `test_scenarios.py`: all 5 scenarios produce valid results

### W6: Visualization
**Commit:** `feat: add visualization (radar, bar, heatmap, salary charts)`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T6.1: Radar Charts | src/hh_competency/viz/radar.py | visual-engineering | programming | T6.2, T6.3 |
| T6.2: Bar/Heatmap | src/hh_competency/viz/bars.py, heatmap.py | visual-engineering | programming | T6.1, T6.3 |
| T6.3: Salary Charts | src/hh_competency/viz/salary.py | visual-engineering | programming | T6.1, T6.2 |

### W7: Methodology Module (runs after W5C)
**Commit:** `feat: add competency-role model construction from validated clusters`

**Depends on:** W5.3 (clustering), W5B (statistical validation), W5C (simulation)

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T7.1: Role Model Builder | src/hh_competency/methodology/role_model.py | deep | programming, m09-domain | T7.2, T7.3 |
| T7.2: Profstandart Mapper | src/hh_competency/methodology/profstandart.py | deep | programming | T7.1, T7.3 |
| T7.3: FGOS Comparator | src/hh_competency/methodology/fgos.py | deep | programming | T7.1, T7.2 |

**T7.1 — Role Model Builder** (`src/hh_competency/methodology/role_model.py`):
- Multi-axis matrix: РАН level (6) × competency axes (from clustering) × proficiency (0–5)
- Role archetype classification: Technician, Researcher, Methodologist, Engineer, Principal, Hybrid
- Cross-specialty role translation matrix (Jaccard overlap between roles)
- Output: `CompetencyMatrix` and `RoleProfile` msgspec models
- Role naming from cluster's top defining skills

**T7.2 — Profstandart Mapper** (`src/hh_competency/methodology/profstandart.py`):
- Load Profstandart 40.011 skill vocabulary (parsed from official XML/JSON)
- Map discovered role clusters to profstandart qualification levels (5–7)
- Identify which discovered roles have NO profstandart coverage (gap signaling)
- Seed vocabulary from FGOS: УК/ОПК/ПК competency lists for cross-referencing

**T7.3 — FGOS Comparator** (`src/hh_competency/methodology/fgos.py`):
- Load FGOS competency lists per specialty code (03.03.02, 04.03.01, etc.)
- Jaccard overlap: how much of FGOS is covered by market demands?
- Market-unique skills: what market wants but FGOS doesn't teach
- FGOS-unique skills: what FGOS teaches but market doesn't demand
- Coverage ratio: |FGOS ∩ Market| / |Market|

**Test requirements:**
- `test_role_model.py`: RoleProfile serialization round-trips, matrix dimensions correct
- `test_profstandart.py`: profstandart vocabulary loads, mapping produces valid levels
- `test_fgos.py`: gap computed correctly for known skill sets, coverage ∈ [0, 1]

### W8: Reports + CLI Wiring
**Commit:** `feat: add analyze/stats/simulate/report CLI commands and main entry point`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T8.1: Analyze CLI | src/hh_competency/commands/analyze.py | quick | programming, domain-cli | T8.2, T8.3, T8.4 |
| T8.2: Stats CLI | src/hh_competency/commands/stats.py | quick | programming, domain-cli | T8.1, T8.3, T8.4 |
| T8.3: Simulate CLI | src/hh_competency/commands/simulate.py | quick | programming, domain-cli | T8.1, T8.2, T8.4 |
| T8.4: Report CLI | src/hh_competency/commands/report.py | quick | programming, domain-cli | T8.1, T8.2, T8.3 |
| T8.5: Main CLI | src/hh_competency/cli.py | quick | programming, domain-cli | after T8.1..T8.4 |

### W9: Integration + Polish
**Commit:** `test: add integration tests, linting pass, and documentation`

| Task | Files | Category | Skills | Parallel |
|------|-------|----------|--------|----------|
| T9.1: Integration Tests | tests/test_integration/ | deep | programming | T9.3 |
| T9.2: Lint + Type | — | quick | programming | T9.1, T9.3 |
| T9.3: README | README.md | writing | — | T9.1 |

## Critical Path
T1 → T2.1 → T2.4 → T3.3 → T4.1 → T4.2 → T5.1 → T5.3 → T5B.1 → T5C.2 → T7.1 → T8.5 → T9.1

## Parallelization Windows
- W2: T2.1, T2.2, T2.3 (parallel) → T2.4
- W3: T3.1, T3.2, T3.3 (all parallel, 3 agents)
- W4+W5: W4 (scrape) parallel with W5 (analysis)
- W5B: T5B.1, T5B.2, T5B.3 (all parallel, 3 agents)
- W5C: T5C.1, T5C.2 (parallel)
- W6: T6.1, T6.2, T6.3 (all parallel, 3 agents)
- W7: T7.1, T7.2, T7.3 (all parallel, 3 agents)
- W8: T8.1, T8.2, T8.3, T8.4 (parallel) → T8.5
- W9: T9.1, T9.3 (parallel) → T9.2

## Verification Per Wave
```bash
W1: uv run ruff check src/
W2: uv run pytest tests/test_storage/ tests/test_nlp/ tests/test_config.py -v
W3: uv run pytest tests/test_storage/ tests/test_nlp/ tests/test_api/ -v
W4: uv run pytest tests/test_cli/test_scrape_cli.py -v
W5: uv run pytest tests/test_analysis/ -v
W5B: uv run pytest tests/test_stats/ -v
W5C: uv run pytest tests/test_simulation/ -v
W6: uv run pytest tests/test_viz/ -v
W7: uv run pytest tests/test_methodology/ -v
W8: uv run pytest tests/test_cli/ -v
W9: uv run ruff check && uv run ty && uv run pytest --cov -v
```

## Dependencies (new)
```toml
# Added to W5B/W5C (scipy, scikit-learn, statsmodels already in pyproject.toml deps)
# No additional dependencies needed beyond those listed in W1 pyproject.toml
```

## MUST NOT DO
- NO plugin architectures or frameworks
- NO web interface/dashboard (CLI only)
- NO NER models (too heavy for skills)
- NO JavaScript frontend
- NO Docker by default
- NO LLM integration (deterministic NLP)
- NO real-time streaming/bots
