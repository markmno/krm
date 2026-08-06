"""ITMO Competency-Role Model Pipeline: end-to-end workflow orchestration.

Chains all phases into a single command for ITMO university curriculum planning:

  hh-competency workflow run --specialties physics,biology,chemistry

Phases:
  1.  Scrape (or skip if data exists)
  2.  NLP + skill extraction (pymorphy3 lemmatization)
  3.  Role discovery (hierarchical clustering)
  4.  Statistical validation (bootstrap significance, chi-squared, ARI)
  5.  Monte Carlo simulation + curriculum optimization (greedy + Pareto)
  6.  Competency matrix (РАН 5-grade system)
  7.  Report generation (HTML + Excel + charts)
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from hh_competency.config import load_config
from hh_competency.storage.db import Database
from hh_competency.storage.models import ScrapeRun

console = Console()
workflow_app = typer.Typer(help="End-to-end ITMO competency pipeline")


# ---------------------------------------------------------------------------
# Main workflow command
# ---------------------------------------------------------------------------


@workflow_app.command(name="run")
def workflow_run(
    specialties: str = typer.Option(
        "all",
        "--specialties",
        "-s",
        help="Specialties to process: 'all' or comma-separated names",
    ),
    config_path: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
    db_path: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    skip_scrape: bool = typer.Option(
        False, "--skip-scrape", help="Skip scraping, use existing data only"
    ),
    max_vacancies: int = typer.Option(
        500, "--max-vacancies", "-n", help="Max vacancies to fetch per specialty"
    ),
    scrape_workers: int = typer.Option(
        3, "--workers", "-w", help="Concurrent detail fetches"
    ),
    n_clusters: int = typer.Option(
        3, "--clusters", "-k", help="Target number of role clusters per specialty"
    ),
    monte_carlo_iter: int = typer.Option(
        500, "--mc-iter", help="Monte Carlo bootstrap iterations"
    ),
    output_dir: str = typer.Option(
        "reports/", "--output-dir", "-o", help="Output directory for reports"
    ),
    export_excel: bool = typer.Option(
        True, "--excel/--no-excel", help="Export Excel workbook with all results"
    ),
    save_json: bool = typer.Option(
        True, "--json/--no-json", help="Save full pipeline results as JSON"
    ),
) -> None:
    """Run the complete ITMO competency-role model pipeline.

    End-to-end workflow: scrape → analyze → validate → simulate → optimize → report.

    Produces:
      - Role clusters per specialty (defining + supporting skills)
      - Statistical significance tests (bootstrap p-values, chi-squared, ARI)
      - Monte Carlo curriculum-fit simulation with confidence intervals
      - Curriculum optimization: minimal skill set for target coverage
      - Competency matrix (РАН 5-grade: мнс→гнс)
      - HTML reports with embedded charts
      - Excel workbook export
      - JSON pipeline results

    Example:
        hh-competency workflow run -s physics,biology -n 300 --mc-iter 1000 -k 4
    """
    # --- Setup ---
    app_config = load_config(config_path)
    spec_names = _resolve_specialty_names(specialties, app_config)
    if not spec_names:
        console.print("[red]No specialties to process[/]")
        raise typer.Exit(code=1)

    db_path = db_path or app_config.database_path
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    console.print(
        Panel.fit(
            f"[bold cyan]ITMO Competency-Role Model Pipeline[/]\n"
            f"Specialties: [yellow]{', '.join(spec_names)}[/]\n"
            f"Database: [dim]{db_path}[/]\n"
            f"Timestamp: {timestamp}",
            title="[bold]Workflow Start[/]",
        )
    )

    results: dict[str, Any] = {
        "pipeline": "itmo-competency",
        "timestamp": timestamp,
        "specialties": spec_names,
        "phases": {},
    }

    # ======================================================================
    # PHASE 1: Scrape
    # ======================================================================
    _run_phase(
        "PHASE 1/9: Data Acquisition",
        _phase_scrape,
        results,
        app_config,
        db_path,
        spec_names,
        max_vacancies,
        scrape_workers,
        skip_scrape,
    )

    # ======================================================================
    # PHASE 2: NLP Skill Extraction
    # ======================================================================
    _run_phase(
        "PHASE 2/10: NLP Skill Extraction",
        _phase_nlp_extract,
        results,
        db_path,
        spec_names,
    )

    # ======================================================================
    # PHASE 2B: Semantic Characteristic Discovery (navec embeddings)
    # ======================================================================
    _run_phase(
        "PHASE 2B/10: Semantic Characteristic Discovery",
        _phase_characteristic_discovery,
        results,
        db_path,
        spec_names,
    )

    # ======================================================================
    # PHASE 3: Role Discovery (Clustering)
    # ======================================================================
    _run_phase(
        "PHASE 3/10: Role Discovery",
        _phase_role_discovery,
        results,
        db_path,
        spec_names,
        n_clusters,
    )

    # ======================================================================
    # PHASE 4: Statistical Validation
    # ======================================================================
    _run_phase(
        "PHASE 4/10: Statistical Validation",
        _phase_statistical_validation,
        results,
        db_path,
        spec_names,
        n_clusters,
    )

    # ======================================================================
    # PHASE 4B: Yearly Trend Analysis
    # ======================================================================
    _run_phase(
        "PHASE 4B/10: Yearly Trend Analysis",
        _phase_trends,
        results,
        db_path,
        spec_names,
    )

    # ======================================================================
    # PHASE 5: Simulation + Optimization
    # ======================================================================
    _run_phase(
        "PHASE 5/10: Monte Carlo Simulation & Curriculum Optimization",
        _phase_simulation,
        results,
        db_path,
        spec_names,
        monte_carlo_iter,
    )

    # ======================================================================
    # PHASE 6: Competency Matrix
    # ======================================================================
    _run_phase(
        "PHASE 6/10: Competency Matrix Construction",
        _phase_competency_matrix,
        results,
        db_path,
        spec_names,
    )

    # ======================================================================
    # PHASE 6B: Mixed/Hybrid Models (STEM+IT, STEM+Eng, STEM+RnD)
    # ======================================================================
    _run_phase(
        "PHASE 6B/10: Hybrid STEM+IT/STEM+Eng/STEM+RnD Models",
        _phase_mixed_models,
        results,
        db_path,
        spec_names,
        app_config,
    )

    # ======================================================================
    # PHASE 7: Reports + Export
    # ======================================================================
    _run_phase(
        "PHASE 7/10: Reports & Export",
        _phase_reports,
        results,
        results,
        db_path,
        spec_names,
        output_dir,
        timestamp,
        export_excel,
        save_json,
        app_config,
    )

    # ======================================================================
    # POST-PHASE 7: Pipeline JSON dump (after _run_phase populated Phase 7)
    # ======================================================================
    if save_json:
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        json_path = os.path.join(data_dir, "pipeline.json")
        try:
            json_results = _sanitize_for_json(results)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2)
            console.print(f"  [green]Pipeline JSON:[/] {json_path}")
        except Exception as exc:
            console.print(f"  [yellow]JSON dump error: {exc}[/]")

    # ======================================================================
    # FINAL: Print pipeline summary
    # ======================================================================
    _print_final_summary(results, output_dir, timestamp)

    console.print(
        Panel.fit(
            f"[bold green]Pipeline Complete[/]\n"
            f"Reports: [cyan]{output_dir}[/]\n"
            f"Timestamp: {timestamp}",
            title="[bold]Workflow Done[/]",
        )
    )


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _run_phase(
    label: str,
    fn: callable,
    results: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Execute a pipeline phase with timing and error handling."""
    console.print(f"\n[bold blue]━━━ {label} ━━━[/]")
    phase_key = label.split(":")[0].strip()
    start = datetime.now(UTC)
    try:
        phase_result = fn(*args, **kwargs)
        elapsed = (datetime.now(UTC) - start).total_seconds()
        console.print(f"  [green]✓[/] Completed in {elapsed:.1f}s")
        results["phases"][phase_key] = {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "data": phase_result,
        }
    except Exception as exc:
        elapsed = (datetime.now(UTC) - start).total_seconds()
        console.print(f"  [red]✗ Failed: {exc}[/]")
        results["phases"][phase_key] = {
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }


def _phase_scrape(
    app_config: Any,
    db_path: str,
    spec_names: list[str],
    skip_scrape: bool,
    max_vacancies: int,
    scrape_workers: int,
) -> dict:
    """Phase 1: Scrape vacancies from HH.ru or use existing data."""
    if skip_scrape:
        console.print("  [yellow]Skipping scrape (--skip-scrape)[/]")
        db = Database(db_path, read_only=False)
        try:
            counts = {}
            for spec in spec_names:
                counts[spec] = db.get_vacancy_count(spec)
            console.print(f"  Existing data: {sum(counts.values())} total vacancies")
            return {"scraped": False, "vacancy_counts": counts}
        finally:
            db.close()

    console.print("  This requires HH.ru API access (OAuth2 token at dev.hh.ru)")
    console.print(
        "  [yellow]If you don't have an API token, use --skip-scrape with existing data[/]"
    )

    from hh_competency.api.client import HHClient
    from hh_competency.commands.scrape import ScrapeEngine

    db = Database(db_path)
    engine = ScrapeEngine(db, app_config)

    async def _run() -> list[ScrapeRun]:
        client = HHClient(
            base_url=app_config.hh_api_base_url,
            user_agent=app_config.hh_api_user_agent,
            requests_per_second=app_config.hh_api_requests_per_second,
            max_retries=app_config.hh_api_max_retries,
            retry_backoff=app_config.hh_api_retry_backoff,
        )
        async with client:
            runs: list[ScrapeRun] = []
            for spec in spec_names:
                run = await engine.scrape_specialty(
                    client, spec, max_vacancies=max_vacancies, workers=scrape_workers
                )
                runs.append(run)
            return runs

    try:
        runs = asyncio.run(_run())
        counts = {r.specialty: r.vacancies_fetched for r in runs}
        total = sum(counts.values())
        console.print(f"  [green]Scraped {total} vacancies across {len(runs)} specialties[/]")
        return {"scraped": True, "vacancy_counts": counts, "total_vacancies": total}
    finally:
        db.close()


def _phase_nlp_extract(
    db_path: str,
    spec_names: list[str],
) -> dict:
    """Phase 2: Extract skills via NLP pipeline (pymorphy3 lemmatization)."""
    from hh_competency.analysis.skills import SkillAnalyzer
    from hh_competency.nlp.pipeline import NLPPipeline

    db = Database(db_path, read_only=False)
    pipeline = NLPPipeline()
    analyzer = SkillAnalyzer(db, pipeline)

    try:
        skill_data: dict[str, dict] = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for spec in spec_names:
                task_id = progress.add_task(
                    f"[cyan]Processing {spec}...", total=None
                )
                try:
                    skills = analyzer.compute_frequencies(spec, min_count=2)
                    skill_data[spec] = {
                        "total_skills": len(skills),
                        "top_skills": [
                            {"lemma": s.lemma, "frequency": s.frequency, "pos": s.pos}
                            for s in skills[:30]
                        ],
                    }
                    progress.update(
                        task_id, visible=False
                    )
                except Exception as exc:
                    skill_data[spec] = {"error": str(exc)}
                    progress.update(task_id, visible=False)

        total_skills = sum(
            d.get("total_skills", 0) for d in skill_data.values()
        )
        console.print(
            f"  Extracted {total_skills} unique skills across {len(spec_names)} specialties"
        )
        return skill_data
    finally:
        db.close()


def _phase_characteristic_discovery(
    db_path: str,
    spec_names: list[str],
) -> dict:
    """Phase 2B: Discover competency characteristics via navec semantic embeddings.

    Uses navec to vectorize skills, then agglomerative clustering on cosine
    similarity to group skills into 3-9 abstract competency characteristics
    (e.g. "Экспериментальные методы", "Научная коммуникация").
    """
    from hh_competency.analysis.skills import SkillAnalyzer
    from hh_competency.nlp.embedder import SkillEmbedder
    from hh_competency.nlp.pipeline import NLPPipeline

    db = Database(db_path, read_only=True)
    try:
        embedder = SkillEmbedder()
        pipeline = NLPPipeline()
        char_data: dict[str, dict] = {}
        total_chars = 0

        for spec in spec_names:
            try:
                analyzer = SkillAnalyzer(db, pipeline)
                skills = analyzer.top_skills(spec, n=100)
                if not skills:
                    char_data[spec] = {"characteristics": [], "total": 0}
                    continue

                skill_lemmas = [s.lemma for s in skills if len(s.lemma) >= 3]

                # Cluster skills semantically
                clusters = embedder.cluster_by_semantics(
                    skill_lemmas,
                    max_clusters=9,
                    min_cluster_size=3,
                    similarity_threshold=0.3,
                )

                chars = []
                for name, members, sim in clusters:
                    chars.append({
                        "name": name,
                        "skills": members[:8],  # top 8 representative skills
                        "coherence": round(float(sim), 3),
                        "size": len(members),
                    })

                char_data[spec] = {
                    "characteristics": chars,
                    "total": len(chars),
                }
                total_chars += len(chars)

                console.print(
                    f"  [cyan]{spec}[/]: {len(chars)} characteristics"
                )
                for c in chars:
                    console.print(
                        f"    [bold]{c['name']}[/] "
                        f"(coherence={c['coherence']:.2f}, "
                        f"skills={c['size']})"
                    )

            except Exception as exc:
                char_data[spec] = {"characteristics": [], "total": 0, "error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        console.print(f"  Total characteristics discovered: {total_chars}")
        return char_data
    finally:
        db.close()


def _phase_role_discovery(
    db_path: str,
    spec_names: list[str],
    n_clusters: int,
) -> dict:
    """Phase 3: Discover role profiles via skill co-occurrence clustering."""
    from hh_competency.analysis.clustering import SkillClusterer

    db = Database(db_path, read_only=False)
    clusterer = SkillClusterer(db)

    try:
        role_data: dict[str, dict] = {}
        total_roles = 0

        for spec in spec_names:
            try:
                roles = clusterer.discover_roles(spec, n_clusters=n_clusters)
                vacancy_names = db.get_vacancy_names(spec)
                role_data[spec] = {
                    "roles_discovered": len(roles),
                    "vacancy_names": vacancy_names,
                    "profiles": [
                        {
                            "role_name": r.role_name,
                            "defining_skills": [
                                {"skill": lemma, "weight": round(w, 4)}
                                for lemma, w in r.defining_skills
                            ],
                            "supporting_skills": [
                                {"skill": lemma, "weight": round(w, 4)}
                                for lemma, w in r.supporting_skills
                            ],
                        }
                        for r in roles
                    ],
                }
                total_roles += len(roles)

                # Print role summary
                console.print(f"  [cyan]{spec}[/]: {len(roles)} role(s)")
                for r in roles:
                    top_def = ", ".join(lemma for lemma, _w in r.defining_skills[:5])
                    console.print(f"    [bold]{r.role_name}[/]: {top_def}")

            except Exception as exc:
                role_data[spec] = {"error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        console.print(f"  Total roles discovered: {total_roles}")
        return role_data
    finally:
        db.close()


def _phase_statistical_validation(
    db_path: str,
    spec_names: list[str],
    n_clusters: int,
) -> dict:
    """Phase 4: Statistical validation of role clusters and differentiation.

    Runs 5 hypothesis tests (H1–H5), each error-isolated so one failing
    test does not block the others:

    H1: Cluster significance — bootstrap permutation test of silhouette score.
    H2: Specialty differentiation — chi-squared test across specialties.
    H3: Experience stratification — chi-squared + ANOVA across experience levels.
    H4: Curriculum-market gap — bootstrap Jaccard to test coverage < 100%.
    H5: Taxonomy stability — bootstrap ARI to test cluster reproducibility.
    """
    from collections import defaultdict

    import numpy as np
    from scipy.stats import chi2_contingency, f_oneway

    from hh_competency.nlp.pipeline import NLPPipeline
    from hh_competency.stats.gaps import GapAnalyzer
    from hh_competency.stats.significance import ClusterSignificanceTester
    from hh_competency.stats.stability import ClusterStabilityTester

    db = Database(db_path, read_only=True)

    try:
        stats_data: dict[str, Any] = {}
        sig_tester = ClusterSignificanceTester(db)
        stab_tester = ClusterStabilityTester(db)

        # ------------------------------------------------------------------
        # H1: Cluster significance — permutation test of silhouette score
        # ------------------------------------------------------------------
        console.print("  [bold]H1: Cluster Significance (permutation test)[/]")
        stats_data["cluster_significance"] = {}
        for spec in spec_names:
            try:
                sig_result = sig_tester.test_cluster_significance(spec, n_iter=100)
                stats_data["cluster_significance"][spec] = {
                    "silhouette_score": sig_result["silhouette_score"],
                    "silhouette_ci_low": sig_result["silhouette_ci_low"],
                    "silhouette_ci_high": sig_result["silhouette_ci_high"],
                    "p_value": sig_result["p_value"],
                }
                p_val = sig_result["p_value"]
                sig_label = (
                    "[bold green]p < 0.01[/]" if p_val < 0.01
                    else "[green]p < 0.05[/]" if p_val < 0.05
                    else f"[yellow]p = {p_val:.4f}[/]"
                )
                console.print(
                    f"  [cyan]{spec}[/]: silhouette={sig_result['silhouette_score']:.3f}, {sig_label}"
                )
            except Exception as exc:
                stats_data["cluster_significance"][spec] = {"error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        # --- Optimal cluster count (auxiliary metric, not a hypothesis) ---
        stats_data["optimal_clusters"] = {}
        for spec in spec_names:
            try:
                cluster_results = sig_tester.test_cluster_count(spec, max_k=6)
                if cluster_results:
                    optimal = next(
                        (r["k"] for r in cluster_results if r.get("is_optimal")),
                        cluster_results[-1]["k"],
                    )
                    stats_data["optimal_clusters"][spec] = {
                        "optimal_k": optimal,
                        "all_k": [
                            {"k": r["k"], "silhouette": r["silhouette_score"]}
                            for r in cluster_results
                        ],
                    }
                    console.print(
                        f"  [cyan]{spec}[/]: optimal k = [bold]{optimal}[/]"
                    )
            except Exception as exc:
                stats_data["optimal_clusters"][spec] = {"error": str(exc)}

        # ------------------------------------------------------------------
        # H2: Specialty differentiation — chi-squared independence test
        # ------------------------------------------------------------------
        console.print("\n  [bold]H2: Specialty Differentiation (chi-squared)[/]")
        if len(spec_names) >= 2:
            try:
                chi2 = stab_tester.test_specialty_differentiation(spec_names)
                stats_data["specialty_differentiation"] = {
                    "chi2_statistic": chi2["chi2_statistic"],
                    "dof": chi2["dof"],
                    "p_value": chi2["p_value"],
                    "cramers_v": chi2["cramers_v"],
                }
                p_val = chi2["p_value"]
                sig_label = (
                    "[bold green]p < 0.01[/]" if p_val < 0.01
                    else "[green]p < 0.05[/]" if p_val < 0.05
                    else f"[yellow]p = {p_val:.4f}[/]"
                )
                console.print(
                    f"  χ²={chi2['chi2_statistic']:.1f}, "
                    f"df={chi2['dof']}, "
                    f"Cramér's V={chi2['cramers_v']:.4f}, "
                    f"{sig_label}"
                )
            except Exception as exc:
                stats_data["specialty_differentiation"] = {"error": str(exc)}
                console.print(f"  [yellow]H2: {exc}[/]")
        else:
            stats_data["specialty_differentiation"] = {
                "error": "Need ≥2 specialties for chi-squared test"
            }

        # ------------------------------------------------------------------
        # H3: Experience stratification
        #     H0: skill profiles do NOT differ by experience level.
        #     Uses chi-squared contingency (experience × top skills) plus
        #     one-way ANOVA on per-vacancy skill-density scores.
        # ------------------------------------------------------------------
        console.print("\n  [bold]H3: Experience Stratification[/]")
        stats_data["experience_stratification"] = {}
        pipeline = NLPPipeline()

        for spec in spec_names:
            try:
                # Query descriptions grouped by experience level
                rows = db.connection.execute(
                    """
                    SELECT
                        COALESCE(
                            NULLIF(json_extract_string(rv.data, '$.experience.name'), ''),
                            'Не указан'
                        ),
                        json_extract_string(rv.data, '$.description')
                    FROM raw_vacancies rv
                    JOIN scrape_runs sr ON rv.run_id = sr.run_id
                    WHERE sr.specialty = ?
                      AND json_extract_string(rv.data, '$.description') IS NOT NULL
                    """,
                    [spec],
                ).fetchall()

                if len(rows) < 10:
                    stats_data["experience_stratification"][spec] = {
                        "error": "Недостаточно вакансий (требуется ≥10)",
                        "n_total": len(rows),
                    }
                    console.print(f"  [yellow]{spec}: всего {len(rows)} вакансий, пропуск[/]")
                    continue

                # Group descriptions by experience level name
                exp_descs: dict[str, list[str]] = defaultdict(list)
                for exp_name, desc in rows:
                    exp_descs[exp_name or "Не указан"].append(desc)

                # Get top skills as column vocabulary
                top_skills = db.get_skill_frequencies(spec, top_n=30)
                skill_vocab = [f.lemma for f in top_skills]
                skill_vocab_set = set(skill_vocab)

                # Build per-experience skill frequency profiles
                exp_profiles: dict[str, dict] = {}
                for exp_name, descs in exp_descs.items():
                    if len(descs) < 3:
                        continue
                    skill_counts: dict[str, int] = defaultdict(int)
                    for desc in descs:
                        keywords = pipeline.extract_keywords(desc)
                        seen: set[str] = set()
                        for lemma, _pos in keywords:
                            if lemma in skill_vocab_set and lemma not in seen:
                                skill_counts[lemma] += 1
                                seen.add(lemma)
                    exp_profiles[exp_name] = {
                        "n_vacancies": len(descs),
                        "skill_counts": skill_counts,
                    }

                if len(exp_profiles) < 2:
                    stats_data["experience_stratification"][spec] = {
                        "error": "Недостаточно уровней опыта с ≥3 вакансиями",
                        "levels_found": list(exp_descs.keys()),
                        "n_total": len(rows),
                    }
                    console.print(f"  [yellow]{spec}: <2 experience levels, пропуск[/]")
                    continue

                valid_levels = sorted(exp_profiles.keys())

                # --- Chi-squared contingency test ---
                table: list[list[int]] = []
                for exp_name in valid_levels:
                    row = [
                        exp_profiles[exp_name]["skill_counts"].get(s, 0)
                        for s in skill_vocab
                    ]
                    table.append(row)

                observed = np.array(table, dtype=int)
                chi2_stat: float | None = None
                p_val: float | None = None
                dof: int | None = None
                cramers_v: float = 0.0

                if observed.sum() >= 10 and observed.shape[0] >= 2 and observed.shape[1] >= 2:
                    try:
                        chi2_val, p_val_val, dof_val, _expected = chi2_contingency(observed)
                        chi2_stat = float(chi2_val)
                        p_val = float(p_val_val)
                        dof = int(dof_val)
                        n = observed.sum()
                        min_dim = min(observed.shape) - 1
                        cramers_v = float(
                            np.sqrt(chi2_stat / (n * min_dim))
                            if n > 0 and min_dim > 0
                            else 0.0
                        )
                    except ValueError:
                        pass

                # --- ANOVA on per-vacancy skill-density score ---
                density_groups: dict[str, list[float]] = defaultdict(list)
                for exp_name, descs in exp_descs.items():
                    if exp_name not in exp_profiles:
                        continue
                    for desc in descs:
                        keywords = pipeline.extract_keywords(desc)
                        seen: set[str] = set()
                        for lemma, _pos in keywords:
                            if lemma in skill_vocab_set:
                                seen.add(lemma)
                        density = len(seen) / len(skill_vocab_set) if skill_vocab_set else 0.0
                        density_groups[exp_name].append(density)

                anova_groups = [
                    np.array(density_groups[name])
                    for name in valid_levels
                    if len(density_groups.get(name, [])) >= 3
                ]
                anova_result: dict | None = None
                if len(anova_groups) >= 2:
                    try:
                        f_stat, anova_p = f_oneway(*anova_groups)
                        anova_result = {
                            "f_statistic": round(float(f_stat), 4),
                            "p_value": round(float(anova_p), 6),
                        }
                    except Exception:
                        anova_result = None

                exp_result: dict[str, Any] = {
                    "experience_levels": valid_levels,
                    "sample_sizes": {
                        name: exp_profiles[name]["n_vacancies"]
                        for name in valid_levels
                    },
                    "interpretation": _h3_interpretation(
                        p_val, cramers_v, valid_levels
                    ),
                }
                if chi2_stat is not None and p_val is not None and dof is not None:
                    exp_result["chi2_statistic"] = round(chi2_stat, 4)
                    exp_result["dof"] = dof
                    exp_result["p_value"] = round(p_val, 6)
                    exp_result["cramers_v"] = round(cramers_v, 4)
                    exp_result["is_significant"] = bool(p_val < 0.05)
                if anova_result:
                    exp_result["anova"] = anova_result

                stats_data["experience_stratification"][spec] = exp_result

                # Console output
                if chi2_stat is not None and p_val is not None:
                    sig_label = (
                        "[bold green]p < 0.01[/]" if p_val < 0.01
                        else "[green]p < 0.05[/]" if p_val < 0.05
                        else f"[yellow]p = {p_val:.4f}[/]"
                    )
                    console.print(
                        f"  [cyan]{spec}[/]: χ²={chi2_stat:.1f}, "
                        f"Cramér's V={cramers_v:.4f}, {sig_label} "
                        f"(уровней: {len(valid_levels)})"
                    )
                else:
                    console.print(
                        f"  [cyan]{spec}[/]: [dim]{len(valid_levels)} групп опыта, "
                        f"недостаточно данных для теста[/]"
                    )

            except Exception as exc:
                stats_data["experience_stratification"][spec] = {"error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        # ------------------------------------------------------------------
        # H4: Curriculum-market gap bootstrap
        #     H0: Jaccard similarity between curriculum and market skill
        #     vocabularies equals 1.0 (no gap). Bootstrap CI tests whether
        #     the true similarity is significantly below 1.0.
        # ------------------------------------------------------------------
        console.print("\n  [bold]H4: Curriculum-Market Gap Bootstrap[/]")
        stats_data["gap_bootstrap"] = {}
        _gap = GapAnalyzer(db)

        for spec in spec_names:
            try:
                # Curriculum = top 25 skills (proxy for taught skills)
                top_curriculum = db.get_skill_frequencies(spec, top_n=25)
                curriculum_set = {f.lemma for f in top_curriculum}

                # Market = top 200 skills (broad demand vocabulary)
                market_skills = db.get_skill_frequencies(spec, top_n=200)
                market_list = list({f.lemma for f in market_skills})

                if len(market_list) < 10:
                    stats_data["gap_bootstrap"][spec] = {
                        "error": "Недостаточно рыночных навыков (требуется ≥10)"
                    }
                    console.print(f"  [yellow]{spec}: <10 market skills, пропуск[/]")
                    continue

                # Bootstrap: resample market skills with replacement
                rng = np.random.default_rng(42)
                n_iter = 200
                jaccards: list[float] = []

                for _ in range(n_iter):
                    sample_size = len(market_list)
                    sampled_idx = rng.choice(sample_size, size=sample_size, replace=True)
                    sampled_set = {market_list[i] for i in sampled_idx}

                    intersection = curriculum_set & sampled_set
                    union = curriculum_set | sampled_set
                    jaccard = len(intersection) / len(union) if union else 0.0
                    jaccards.append(jaccard)

                arr = np.array(jaccards)
                ci_low = float(np.percentile(arr, 2.5))
                ci_high = float(np.percentile(arr, 97.5))
                mean_jaccard = float(np.mean(arr))

                # Significant gap ≡ 95% CI is entirely below 1.0
                is_significant = ci_high < 1.0

                stats_data["gap_bootstrap"][spec] = {
                    "mean_jaccard": round(mean_jaccard, 4),
                    "ci_low": round(ci_low, 4),
                    "ci_high": round(ci_high, 4),
                    "is_significant": is_significant,
                    "curriculum_size": len(curriculum_set),
                    "market_size": len(market_list),
                    "interpretation": (
                        f"Сходство Жаккара учебной программы и рынка: {mean_jaccard:.3f} "
                        f"[{ci_low:.3f}, {ci_high:.3f}]. "
                    ) + (
                        "Значимый разрыв: 95% ДИ не включает 1.0."
                        if is_significant
                        else (
                            "Разрыв незначим: 95% ДИ включает 1.0 "
                            "(нет статистического подтверждения разрыва)."
                        )
                    ),
                }

                gap_label = (
                    "[bold green]значимый разрыв[/]" if is_significant
                    else "[yellow]разрыв незначим[/]"
                )
                console.print(
                    f"  [cyan]{spec}[/]: Jaccard={mean_jaccard:.4f} "
                    f"CI=[{ci_low:.4f}, {ci_high:.4f}], {gap_label}"
                )

            except Exception as exc:
                stats_data["gap_bootstrap"][spec] = {"error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        # ------------------------------------------------------------------
        # H5: Taxonomy stability — bootstrap Adjusted Rand Index
        #     H0: cluster assignments are unstable (ARI ≈ 0).
        #     Bootstrap subsamples → recluster → compare ARI to full-sample
        #     clustering. High ARI (>0.7) = stable taxonomy.
        # ------------------------------------------------------------------
        console.print("\n  [bold]H5: Taxonomy Stability (Bootstrap ARI)[/]")
        stats_data["taxonomy_stability"] = {}

        for spec in spec_names:
            try:
                ari_result = stab_tester.bootstrap_rand_index(
                    specialty=spec,
                    n_iter=50,
                    sample_frac=0.8,
                    n_clusters=n_clusters,
                )

                ari_mean = ari_result["rand_index_mean"]
                stats_data["taxonomy_stability"][spec] = {
                    "rand_index_mean": round(ari_mean, 4),
                    "rand_index_ci_low": ari_result["rand_index_ci_low"],
                    "rand_index_ci_high": ari_result["rand_index_ci_high"],
                    "n_iter": ari_result["n_iter"],
                    "interpretation": _h5_interpretation(ari_mean),
                }

                stability_label = (
                    "[bold green]высокая стабильность[/]" if ari_mean >= 0.7
                    else "[green]умеренная[/]" if ari_mean >= 0.4
                    else "[yellow]низкая[/]"
                )
                console.print(
                    f"  [cyan]{spec}[/]: ARI={ari_mean:.4f} "
                    f"CI=[{ari_result['rand_index_ci_low']:.4f}, "
                    f"{ari_result['rand_index_ci_high']:.4f}], "
                    f"{stability_label} (n={ari_result['n_iter']})"
                )

            except Exception as exc:
                stats_data["taxonomy_stability"][spec] = {"error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        return stats_data
    finally:
        db.close()


def _phase_trends(
    db_path: str,
    spec_names: list[str],
) -> dict:
    """Phase 4B: Yearly skill trend analysis with Mann-Kendall test."""
    from hh_competency.analysis.trends import analyze_trends
    from hh_competency.storage.db import Database

    db = Database(db_path, read_only=True)
    try:
        trends_data: dict[str, Any] = {}

        for spec in spec_names:
            try:
                trend_result = analyze_trends(db, spec, min_count=2)
                trends_data[spec] = trend_result

                total = trend_result.get("total_years", 0)
                emerging = len(trend_result.get("emerging", []))
                declining = len(trend_result.get("declining", []))
                stable = len(trend_result.get("stable", []))
                verdict = trend_result.get("sustainability_verdict", "insufficient data")

                console.print(
                    f"  [cyan]{spec}[/]: {total} years, "
                    f"↑{emerging} ↓{declining} →{stable} — {verdict}"
                )
            except Exception as exc:
                trends_data[spec] = {"error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        return trends_data
    finally:
        db.close()


def _phase_simulation(
    db_path: str,
    spec_names: list[str],
    mc_iter: int,
) -> dict:
    """Phase 5: Monte Carlo simulation + curriculum optimization."""
    from hh_competency.analysis.skills import SkillAnalyzer
    from hh_competency.nlp.pipeline import NLPPipeline
    from hh_competency.simulation.monte_carlo import MonteCarloEngine
    from hh_competency.simulation.optimizer import CurriculumOptimizer

    db = Database(db_path, read_only=True)
    pipeline = NLPPipeline()
    analyzer = SkillAnalyzer(db, pipeline)
    mc_engine = MonteCarloEngine(db, pipeline)
    optimizer = CurriculumOptimizer(db, pipeline)

    try:
        sim_data: dict[str, Any] = {}

        for spec in spec_names:
            console.print(f"\n  [bold cyan]{spec}[/]")

            # Get top skills as curriculum baseline
            try:
                top_skills = analyzer.top_skills(spec, n=25)
                curriculum = [s.lemma for s in top_skills]
            except Exception:
                curriculum = []
                console.print("    [yellow]No skill data for simulation[/]")
                continue

            spec_result: dict[str, Any] = {}

            # 5a. Monte Carlo curriculum fit
            try:
                mc_result = mc_engine.simulate_curriculum_fit(
                    curriculum_skills=curriculum,
                    specialty=spec,
                    n_iter=mc_iter,
                )
                spec_result["monte_carlo"] = {
                    "scenario": mc_result.scenario_name,
                    "mean_coverage": mc_result.mean_coverage,
                    "ci_lower": mc_result.ci_lower,
                    "ci_upper": mc_result.ci_upper,
                    "n_samples": mc_result.n_samples,
                    "skill_count": len(mc_result.skill_set),
                }
                console.print(
                    f"    MC fit: mean={mc_result.mean_coverage*100:.1f}% "
                    f"CI=[{mc_result.ci_lower*100:.1f}%, {mc_result.ci_upper*100:.1f}%]"
                )
            except Exception as exc:
                spec_result["monte_carlo"] = {"error": str(exc)}
                console.print(f"    [yellow]MC error: {exc}[/]")

            # 5b. Greedy curriculum optimization
            try:
                greedy = optimizer.greedy_optimize(
                    specialty=spec,
                    max_skills=40,
                    threshold=0.8,
                )
                spec_result["optimization"] = {
                    "method": "greedy",
                    "selected_skills": greedy["selected_skills"],
                    "final_coverage": greedy["final_coverage"],
                    "skills_count": len(greedy["selected_skills"]),
                }
                console.print(
                    f"    Greedy: {len(greedy['selected_skills'])} skills → "
                    f"{greedy['final_coverage']*100:.1f}% coverage"
                )
            except Exception as exc:
                spec_result["optimization"] = {"error": str(exc)}
                console.print(f"    [yellow]Optimizer error: {exc}[/]")

            # 5c. Pareto frontier (for visualization)
            try:
                pareto = optimizer.pareto_frontier(specialty=spec, max_skills=30)
                spec_result["pareto_frontier"] = pareto
                if pareto:
                    last = pareto[-1]
                    console.print(
                        f"    Pareto: k={last['k']} → {last['coverage']*100:.1f}% coverage"
                    )
            except Exception as exc:
                spec_result["pareto_frontier"] = {"error": str(exc)}

            sim_data[spec] = spec_result

        return sim_data
    finally:
        db.close()


def _phase_competency_matrix(
    db_path: str,
    spec_names: list[str],
) -> dict:
    """Phase 6: Build competency matrix (РАН 5-grade system)."""
    from hh_competency.methodology.role_builder import CompetencyRoleBuilder

    db = Database(db_path, read_only=False)
    builder = CompetencyRoleBuilder(db)

    try:
        matrix_data: dict[str, dict] = {}
        for spec in spec_names:
            try:
                model = builder.build_role_model(spec)
                matrix = model["компетентностная_матрица"]
                matrix_data[spec] = {
                    "role_profiles": len(model["ролевые_профили"]),
                    "axes": [
                        {"name": a["название"], "skills": a["типовые_навыки"]}
                        for a in matrix["оси"]
                    ],
                    "levels": matrix["уровни"],
                    "proficiency_matrix": matrix["матрица_владения"],
                }

                # Print matrix summary
                console.print(f"  [cyan]{spec}[/]: {len(matrix['оси'])} axes × {len(matrix['уровни'])} levels")
                for i, axis in enumerate(matrix["оси"]):
                    # Proficiency matrix is [level][axis] — extract column i for all levels
                    level_values = [
                        matrix["матрица_владения"][j][i]
                        if j < len(matrix["матрица_владения"]) and i < len(matrix["матрица_владения"][j])
                        else 0
                        for j in range(len(matrix["уровни"]))
                    ]
                    console.print(
                        f"    [{axis['название']}]: "
                        + " → ".join(str(v) for v in level_values)
                    )

            except Exception as exc:
                matrix_data[spec] = {"error": str(exc)}
                console.print(f"  [yellow]{spec}: {exc}[/]")

        return matrix_data
    finally:
        db.close()


def _phase_mixed_models(
    db_path: str,
    spec_names: list[str],
    app_config: Any,
) -> dict:
    """Phase 6B: Generate hybrid STEM+IT, STEM+Engineering, STEM+RnD models."""
    from hh_competency.methodology.mixed_models import MixedModelBuilder
    from hh_competency.storage.db import Database

    db = Database(db_path, read_only=True)
    builder = MixedModelBuilder(db, app_config)

    try:
        mixed_data: dict[str, Any] = {}

        hybrid_specs = []
        for spec_name in spec_names:
            spec = app_config.specialties.get(spec_name)
            if spec and spec.mix_of and len(spec.mix_of) >= 2:
                hybrid_specs.append(spec_name)

        if not hybrid_specs:
            console.print("  [dim]No hybrid specialties with mix_of defined — skipping[/]")
            return {}

        for spec_name in hybrid_specs:
            try:
                profile = builder.build_mixed_profile(
                    mixed_name=spec_name,
                    strategy="weighted",
                )
                mixed_data[spec_name] = {
                    "source_specialties": profile.parents,
                    "total_unique_skills": profile.total_unique_skills,
                    "shared_skill_count": profile.shared_skill_count,
                    "jaccard_similarity": profile.jaccard_similarity,
                    "uniqueness_map": profile.uniqueness_map,
                    "source_distribution": profile.source_distribution,
                }
                console.print(
                    f"  [cyan]{spec_name}[/]: {profile.total_unique_skills} skills "
                    f"(shared={profile.shared_skill_count}, Jaccard={profile.jaccard_similarity:.3f}, "
                    f"sources={profile.source_distribution})"
                )
            except Exception as exc:
                mixed_data[spec_name] = {"error": str(exc)}
                console.print(f"  [yellow]{spec_name}: {exc}[/]")

        return mixed_data
    finally:
        db.close()


def _phase_reports(
    results: dict[str, Any],
    db_path: str,
    spec_names: list[str],
    output_dir: str,
    timestamp: str,
    export_excel: bool,
    save_json: bool,
    app_config: Any,
) -> dict:
    """Phase 7: Generate reports (HTML, Excel, JSON, charts)."""
    import pandas as pd

    from hh_competency.nlp.pipeline import NLPPipeline
    from hh_competency.storage.db import Database
    from hh_competency.viz.bars import generate_role_profile_chart, generate_skill_bar_chart
    from hh_competency.viz.heatmap import generate_overlap_heatmap
    from hh_competency.viz.radar import generate_radar_chart
    from hh_competency.viz.salary import generate_salary_boxplot

    db = Database(db_path, read_only=False)  # Writes: Excel, abstractor, discover_roles
    pipeline = NLPPipeline()

    try:
        report_data: dict[str, Any] = {}

        # --- 7a. Abstract characteristics (3-9 per specialty) ---
        console.print("  [dim]Computing abstract competency characteristics...[/]")
        try:
            from hh_competency.analysis.abstractor import CompetencyAbstractor

            abstractor = CompetencyAbstractor(db)
            abstract_chars: dict[str, list[dict]] = {}
            pd.options.mode.chained_assignment = None
            for spec in spec_names:
                try:
                    abstraction = abstractor.discover(spec)
                    chars_list = [
                        {
                            "name": c.name,
                            "category": c.category,
                            "representative_skills": c.representative_skills,
                            "coverage_pct": c.coverage_pct,
                            "stability": c.stability_score,
                            "is_core": c.is_core,
                        }
                        for c in abstraction.characteristics
                    ]
                    abstract_chars[spec] = chars_list
                    console.print(
                        f"    [{spec}]: {len(chars_list)} characteristics "
                        f"({sum(1 for c in chars_list if c['is_core'])} core)"
                    )
                except Exception as exc:
                    console.print(f"    [yellow]{spec}: {exc}[/]")
                    abstract_chars[spec] = []
        except Exception as exc:
            console.print(f"  [yellow]Abstractor: {exc}[/]")
            abstract_chars = {}

        # --- 7b. Restructured output hierarchy ---
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        charts_dir = os.path.join(run_dir, "charts")
        html_dir = os.path.join(run_dir, "reports")
        data_dir = os.path.join(run_dir, "data")
        for d in [charts_dir, html_dir, data_dir]:
            os.makedirs(d, exist_ok=True)

        # --- 7c. Generate charts ---
        console.print("  [dim]Generating charts...[/]")

        # Radar chart (cross-specialty)
        if len(spec_names) >= 2:
            radar_path = os.path.join(charts_dir, "radar.html")
            try:
                generate_radar_chart(db, spec_names, top_n_skills=8, output_path=radar_path)
                report_data["radar_chart"] = radar_path
                console.print(f"    Radar: {radar_path}")
            except Exception as exc:
                console.print(f"    [yellow]Radar error: {exc}[/]")

        # Overlap heatmap
        if len(spec_names) >= 2:
            heatmap_path = os.path.join(charts_dir, "overlap_heatmap.html")
            try:
                generate_overlap_heatmap(db, spec_names, output_path=heatmap_path)
                report_data["overlap_heatmap"] = heatmap_path
                console.print(f"    Overlap heatmap: {heatmap_path}")
            except Exception as exc:
                console.print(f"    [yellow]Heatmap error: {exc}[/]")

        # Salary boxplot (all specialties)
        try:
            salary_path = os.path.join(charts_dir, "salary.html")
            generate_salary_boxplot(db, spec_names, output_path=salary_path)
            report_data["salary_boxplot"] = salary_path
            console.print(f"    Salary: {salary_path}")
        except Exception as exc:
            console.print(f"    [yellow]Salary chart: {exc}[/]")

        # Per-specialty charts
        for spec in spec_names:
            try:
                bar_path = os.path.join(charts_dir, f"skills_{spec}.html")
                generate_skill_bar_chart(db, spec, top_n=20, output_path=bar_path)
                report_data.setdefault("skill_bars", {})[spec] = bar_path
            except Exception:
                console.print(
                    f"    [yellow]Skill bar chart for {spec} failed[/]"
                )

            try:
                role_path = os.path.join(charts_dir, f"roles_{spec}.html")
                generate_role_profile_chart(db, spec, output_path=role_path)
                report_data.setdefault("role_charts", {})[spec] = role_path
            except Exception:
                console.print(
                    f"    [yellow]Role profile chart for {spec} failed[/]"
                )

        # --- 7d. Excel export ---
        if export_excel:
            console.print("  [dim]Exporting Excel workbook...[/]")
            try:
                xlsx_path = _export_excel_workbook(
                    db, pipeline, spec_names, data_dir, timestamp, results
                )
                report_data["excel"] = xlsx_path
                console.print(f"    Excel: {xlsx_path}")
            except Exception as exc:
                console.print(f"    [yellow]Excel export error: {exc}[/]")

        # --- 7e. HTML reports per specialty ---
        console.print("  [dim]Generating HTML reports...[/]")
        report_data["abstract_characteristics"] = abstract_chars
        for spec in spec_names:
            try:
                html_path = _generate_html_report(
                    db, pipeline, spec, html_dir, timestamp, results=results
                )
                report_data.setdefault("html_reports", {})[spec] = html_path
                console.print(f"    HTML [{spec}]: {html_path}")
            except Exception as exc:
                console.print(f"    [yellow]HTML [{spec}]: {exc}[/]")

        return report_data
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _export_excel_workbook(
    db: Database,
    pipeline: Any,
    spec_names: list[str],
    output_dir: str,
    timestamp: str,
    results: dict[str, Any],
) -> str:
    """Export full pipeline results as multi-sheet Excel workbook."""
    from hh_competency.analysis.clustering import SkillClusterer

    clusterer = SkillClusterer(db)

    try:
        import openpyxl
    except ImportError:
        raise ImportError(
            "openpyxl not installed. Run: uv add openpyxl"
        ) from None

    wb = openpyxl.Workbook()

    # Sheet 1: Overview
    ws = wb.active
    ws.title = "Обзор"
    ws.append(["hh-competency Pipeline Results"])
    ws.append(["Timestamp", timestamp])
    ws.append(["Specialties", ", ".join(spec_names)])
    ws.append([])
    ws.append(["Специальность", "Вакансий", "Навыков", "Ролей", "Покрытие (MC)", "Оптимальное k"])

    for spec in spec_names:
        vc = db.get_vacancy_count(spec)
        skills = db.get_skill_frequencies(spec, top_n=500)
        try:
            roles = clusterer.discover_roles(spec)
            rc = len(roles)
        except Exception as exc:
            console.print(f"    [yellow]Role discovery failed for {spec}: {exc}[/]")
            rc = 0

        mc_cov = "N/A"
        phases = results.get("phases", {})
        sim_phase = phases.get("PHASE 5/10", {})
        if sim_phase.get("status") == "ok":
            sim_data = sim_phase.get("data", {}).get(spec, {})
            mc = sim_data.get("monte_carlo", {})
            if isinstance(mc, dict) and "mean_coverage" in mc:
                mc_cov = f"{mc['mean_coverage']*100:.1f}%"

        k_opt = "N/A"
        stats_phase = phases.get("PHASE 4/10", {})
        if stats_phase.get("status") == "ok":
            stats_data = stats_phase.get("data", {}).get("optimal_clusters", {})
            k_info = stats_data.get(spec, {})
            if isinstance(k_info, dict) and "optimal_k" in k_info:
                k_opt = k_info["optimal_k"]

        ws.append([spec, vc, len(skills), rc, mc_cov, k_opt])

    # Sheet 2: Skills
    ws_skills = wb.create_sheet("Навыки")
    ws_skills.append(["Специальность", "Лемма", "POS", "Частота", "Вакансий"])
    for spec in spec_names:
        try:
            skills = db.get_skill_frequencies(spec, top_n=200)
            for s in skills:
                ws_skills.append([s.specialty, s.lemma, s.pos, s.frequency, s.vacancy_count])
        except Exception as exc:
            console.print(f"  [yellow]Excel skills export failed for {spec}: {exc}[/]")

    # Sheet 3: Roles
    ws_roles = wb.create_sheet("Роли")
    ws_roles.append(["Специальность", "Роль", "Определяющие навыки", "Поддерживающие навыки"])
    for spec in spec_names:
        try:
            roles = clusterer.discover_roles(spec)
            for r in roles:
                def_skills = "; ".join(f"{lemma}({w:.2f})" for lemma, w in r.defining_skills)
                sup_skills = "; ".join(f"{lemma}({w:.2f})" for lemma, w in r.supporting_skills)
                ws_roles.append([spec, r.role_name, def_skills, sup_skills])
        except Exception:
            pass

    # Sheet 4: Simulation
    ws_sim = wb.create_sheet("Симуляция")
    ws_sim.append(["Специальность", "Метод", "Покрытие", "CI нижн", "CI верх", "Навыков"])
    phases = results.get("phases", {})
    sim_phase = phases.get("PHASE 5/10", {})
    if sim_phase.get("status") == "ok":
        for spec, data in sim_phase.get("data", {}).items():
            mc = data.get("monte_carlo", {})
            if isinstance(mc, dict) and "mean_coverage" in mc:
                ws_sim.append([
                    spec, "Monte Carlo",
                    f"{mc['mean_coverage']*100:.1f}%",
                    f"{mc['ci_lower']*100:.1f}%",
                    f"{mc['ci_upper']*100:.1f}%",
                    mc.get("skill_count", 0),
                ])
            opt = data.get("optimization", {})
            if isinstance(opt, dict) and "final_coverage" in opt:
                ws_sim.append([
                    spec, "Greedy",
                    f"{opt['final_coverage']*100:.1f}%",
                    "N/A", "N/A",
                    opt.get("skills_count", 0),
                ])

    # Sheet 5: Competency Matrix
    ws_matrix = wb.create_sheet("Матрица компетенций")
    ws_matrix.append(["hh-competency Competency Matrix", "", "", "", "", "", ""])
    ws_matrix.append(["Специальность", "Ось", "мнс", "нс", "снс", "внс", "гнс"])
    matrix_phase = phases.get("PHASE 6/10", {})
    if matrix_phase.get("status") == "ok":
        for spec, data in matrix_phase.get("data", {}).items():
            axes = data.get("axes", [])
            pmatrix = data.get("proficiency_matrix", [])
            for i, axis in enumerate(axes):
                row_vals = [spec, axis["name"]]
                if i < len(pmatrix):
                    row_vals.extend(str(v) for v in pmatrix[i])
                else:
                    row_vals.extend([""] * 5)
                ws_matrix.append(row_vals)

    xlsx_path = os.path.join(output_dir, f"pipeline_{timestamp}.xlsx")
    wb.save(xlsx_path)
    return xlsx_path


def _generate_html_report(
    db: Database,
    pipeline: Any,
    specialty: str,
    output_dir: str,
    timestamp: str,
    results: dict[str, Any] | None = None,
) -> str:
    """Generate a role-centric HTML report with spider charts per role.

    Structure:
    - Page 1: Market overview (yearly dynamics, shared statistics, salary)
    - Per role: spider chart of competency characteristics, defining/supporting skills, statistics
    """
    import json as _json

    from hh_competency.viz.bars import generate_skill_bar_chart
    from hh_competency.viz.salary import generate_salary_boxplot

    # Collect data
    vacancy_count = db.get_vacancy_count(specialty)
    top_skills = db.get_skill_frequencies(specialty, top_n=50)
    try:
        role_profiles = db.get_role_profiles(specialty)
    except Exception:
        role_profiles = []

    # Extract abstract characteristics, matrix data, stats, and MC data from results
    abstract_chars: list[dict] = []
    matrix_data: dict = {}
    stats_data: dict = {}
    mc_data: dict = {}
    if results:
        phases = results.get("phases", {})
        # Phase 6: Competency Matrix
        matrix_phase = phases.get("PHASE 6/10", {}).get("data", {})
        if isinstance(matrix_phase, dict):
            matrix_data = matrix_phase.get(specialty, {})
            if isinstance(matrix_data, dict):
                chars = matrix_data.get("abstract_characteristics", [])
                if isinstance(chars, list):
                    abstract_chars = [
                        {
                            "name": c.get("name", c) if isinstance(c, dict) else str(c),
                            "category": c.get("category", "") if isinstance(c, dict) else "",
                            "coverage_pct": c.get("coverage_pct", 0) if isinstance(c, dict) else 0,
                            "is_core": c.get("is_core", False) if isinstance(c, dict) else False,
                            "representative_skills": c.get("representative_skills", []) if isinstance(c, dict) else [],
                        }
                        for c in chars
                    ]
        # Phase 4: Statistical Validation
        stats_phase_data = phases.get("PHASE 4/10", {}).get("data", {})
        if isinstance(stats_phase_data, dict):
            stats_data = stats_phase_data
            if not isinstance(stats_data, dict):
                stats_data = {}
        # Phase 5: Monte Carlo Simulation
        mc_phase_data = phases.get("PHASE 5/10", {}).get("data", {})
        if isinstance(mc_phase_data, dict):
            mc_data = mc_phase_data.get(specialty, {})
            if not isinstance(mc_data, dict):
                mc_data = {}

    # Vacancy names for this specialty
    try:
        vacancy_names = db.get_vacancy_names(specialty)
    except Exception:
        vacancy_names = []

    # ── Build role sections ──
    role_sections_html = ""
    for role in role_profiles:
        try:
            role_name = role.role_name
            defining = (
                _json.loads(role.defining_skills)
                if isinstance(role.defining_skills, str)
                else role.defining_skills
            )
            supporting = (
                _json.loads(role.supporting_skills)
                if isinstance(role.supporting_skills, str)
                else role.supporting_skills
            )

            # Spider chart: competency characteristics for this role
            spider_html = _generate_role_spider_chart(role_name, abstract_chars)

            # Skills table
            def_rows = "".join(
                f"<tr><td>{s[0] if isinstance(s, (list, tuple)) else s}</td>"
                f"<td>{s[1] if isinstance(s, (list, tuple)) and len(s) > 1 else '—'}</td></tr>"
                for s in defining[:15]
            )
            sup_rows = "".join(
                f"<tr><td>{s[0] if isinstance(s, (list, tuple)) else s}</td>"
                f"<td>{s[1] if isinstance(s, (list, tuple)) and len(s) > 1 else '—'}</td></tr>"
                for s in supporting[:15]
            )

            role_sections_html += f"""
<div class="role-section" id="role-{role_name.replace(' ', '-')}">
<h2>🔹 {role_name}</h2>
<div class="vacancy-titles">Вакансии: {", ".join(vacancy_names[:10]) if vacancy_names else "—"}</div>
{spider_html}
<div class="skills-grid">
  <div>
    <h3>Определяющие навыки</h3>
    <table>
      <thead><tr><th>Навык</th><th>Вес</th></tr></thead>
      <tbody>{def_rows or '<tr><td colspan="2">—</td></tr>'}</tbody>
    </table>
  </div>
  <div>
    <h3>Поддерживающие навыки</h3>
    <table>
      <thead><tr><th>Навык</th><th>Вес</th></tr></thead>
      <tbody>{sup_rows or '<tr><td colspan="2">—</td></tr>'}</tbody>
    </table>
  </div>
</div>
</div>
"""
        except Exception:
            pass

    # ── Market overview section (top) ──
    try:
        skills_fig = generate_skill_bar_chart(db, specialty, top_n=20)
        skills_html = _fig_to_html_iframe(skills_fig)
    except Exception:
        skills_html = "<p>Недостаточно данных</p>"

    try:
        salary_fig = generate_salary_boxplot(db, [specialty])
        salary_html = _fig_to_html_iframe(salary_fig)
    except Exception:
        salary_html = "<p>Нет данных о зарплатах</p>"

    # Skills table (top 30)
    skills_rows = ""
    for s in top_skills[:30]:
        skills_rows += (
            f"<tr><td>{s.lemma}</td><td>{s.pos}</td>"
            f"<td>{s.frequency}</td><td>{s.vacancy_count}</td></tr>\n"
        )

    # Abstract characteristics summary
    chars_cards = ""
    for c in abstract_chars:
        core_badge = ' <span class="core-badge">ЯДРО</span>' if c.get("is_core") else ""
        reps = ", ".join(c.get("representative_skills", [])[:5])
        chars_cards += f"""
<div class="stat-card">
  <div class="value">{c.get('coverage_pct', 0):.0f}%{core_badge}</div>
  <div class="label">{c['name']}</div>
  <div class="detail">{reps}</div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hh-competency: {specialty}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; background: #fff; }}
  h1 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; margin-top: 40px; }}
  h3 {{ color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #1f77b4; color: #fff; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
  .nav {{ position: sticky; top: 0; background: #fff; padding: 8px 0;
          border-bottom: 1px solid #eee; margin-bottom: 24px; z-index: 10; }}
  .nav a {{ margin-right: 16px; color: #1f77b4; text-decoration: none; }}
  .nav a:hover {{ text-decoration: underline; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
  .stat-card {{ background: #f0f4f8; padding: 16px; border-radius: 8px; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: bold; color: #1f77b4; }}
  .stat-card .label {{ color: #666; font-size: 0.9em; }}
  .stat-card .detail {{ font-size: 0.8em; color: #999; margin-top: 4px; }}
  .core-badge {{ background: #1f77b4; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; }}
  .skills-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .role-section {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 24px; margin: 32px 0; }}
  .role-section h2 {{ margin-top: 0; color: #1f77b4; }}
  .chart-container {{ margin: 24px 0; }}
  @media (max-width: 768px) {{
    .skills-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>hh-competency: {specialty}</h1>
<div class="meta">
  Сгенерировано: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
  Вакансий в базе: {vacancy_count} | Уникальных навыков: {len(top_skills)} | Ролей: {len(role_profiles)}<br>
  Частые названия вакансий: {", ".join(vacancy_names[:10]) if vacancy_names else "—"}
</div>

<!-- Navigation -->
<div class="nav">
  <a href="#overview">📊 Обзор рынка</a>
  <a href="#characteristics">🎯 Характеристики</a>
  <a href="#skills">📋 Топ навыков</a>
  <a href="#matrix">📊 Матрица РАН</a>
  <a href="#stats">📈 Статистика H1-H5</a>
  <a href="#monte-carlo">🎲 Монте-Карло</a>"""

    for role in role_profiles:
        try:
            safe_id = role.role_name.replace(" ", "-")
            html += f'\n  <a href="#role-{safe_id}">🔹 {role.role_name}</a>'
        except Exception:
            pass
    html += '\n</div>\n'

    # Section 1: Market overview
    html += f"""
<div id="overview">
<h2>📊 Обзор рынка</h2>
<div class="stats-grid">
  <div class="stat-card">
    <div class="value">{vacancy_count}</div>
    <div class="label">Вакансий</div>
  </div>
  <div class="stat-card">
    <div class="value">{len(top_skills)}</div>
    <div class="label">Уникальных навыков</div>
  </div>
  <div class="stat-card">
    <div class="value">{len(role_profiles)}</div>
    <div class="label">Ролевых профилей</div>
  </div>
</div>
<h3>Заработная плата</h3>
<div class="chart-container">{salary_html}</div>
</div>

<!-- Section 2: Competency characteristics -->
<div id="characteristics">
<h2>🎯 Компетентностные характеристики</h2>
<div class="stats-grid">{chars_cards or '<p>Недостаточно данных для извлечения характеристик</p>'}</div>
</div>

<!-- Section 3: Top skills -->
<div id="skills">
<h2>📋 Топ-30 навыков рынка</h2>
<div class="chart-container">{skills_html}</div>
<table>
  <thead><tr><th>Навык</th><th>POS</th><th>Частота</th><th>Вакансий</th></tr></thead>
  <tbody>{skills_rows}</tbody>
</table>
</div>
"""

    # Section 4+: Per-role pages
    html += role_sections_html

    # ── Section: Competency Matrix (РАН) ──
    axes = matrix_data.get("axes", []) if matrix_data else []
    pmatrix = matrix_data.get("proficiency_matrix", []) if matrix_data else []
    if axes:
        level_abbrevs = ["мнс", "нс", "снс", "внс", "гнс"]
        matrix_rows = ""
        num_levels = len(matrix_data.get("levels", [])) or 5
        for i, axis in enumerate(axes):
            axis_name = axis["name"] if isinstance(axis, dict) else str(axis)
            matrix_rows += f"<tr><td>{axis_name}</td>"
            # proficiency_matrix is [level][axis] — transpose by column
            for j in range(num_levels):
                try:
                    val = pmatrix[j][i]
                    matrix_rows += f"<td>{val}</td>"
                except (IndexError, TypeError):
                    matrix_rows += "<td>—</td>"
            matrix_rows += "</tr>\n"
        html += f"""
<!-- Section: Competency Matrix (РАН) -->
<div id="matrix">
<h2>📊 Компетентностно-ролевая модель (РАН)</h2>
<p>Матрица компетенций по 5-уровневой системе ({", ".join(level_abbrevs)}): младший научный сотрудник → главный научный сотрудник</p>
<table>
  <thead><tr><th>Ось компетенций</th><th>мнс</th><th>нс</th><th>снс</th><th>внс</th><th>гнс</th></tr></thead>
  <tbody>{matrix_rows}</tbody>
</table>
</div>
"""
    else:
        html += '<div id="matrix"><h2>📊 Компетентностно-ролевая модель (РАН)</h2><p>Недостаточно данных</p></div>'

    # ── Section: Statistical Validation (H1-H5) ──
    # Phase 4 data is nested: {metric_name: {specialty: data}}
    # specialty_differentiation (H2) is cross-specialty, not nested by spec
    h1_data = (stats_data.get("cluster_significance") or {}).get(specialty) if stats_data else None
    h2_data = (stats_data.get("specialty_differentiation") or {}) if stats_data else None
    h3_data = (stats_data.get("experience_stratification") or {}).get(specialty) if stats_data else None
    h4_data = (stats_data.get("gap_bootstrap") or {}).get(specialty) if stats_data else None
    h5_data = (stats_data.get("taxonomy_stability") or {}).get(specialty) if stats_data else None

    has_any_stats = bool(h1_data or h2_data or h3_data or h4_data or h5_data)

    if has_any_stats:
        def _fmt_pval(p: float | None) -> str:
            if p is None:
                return "—"
            if p < 0.0001:
                return "&lt;0.0001"
            return f"{p:.4f}"

        def _verdict_from_pval(p: float | None) -> str:
            if p is None:
                return "Недостаточно данных"
            if p < 0.01:
                return "p &lt; 0.01 — значимо"
            if p < 0.05:
                return "p &lt; 0.05 — значимо"
            return f"p = {p:.4f} — не значимо"

        # ── H1: Cluster Significance (silhouette permutation test) ──
        if h1_data:
            h1_score = h1_data.get("silhouette_score")
            h1_p = h1_data.get("p_value")
            h1_lo = h1_data.get("silhouette_ci_low")
            h1_hi = h1_data.get("silhouette_ci_high")
            h1_verdict = _verdict_from_pval(h1_p)
            h1_detail = ""
            if h1_score is not None:
                h1_detail += f"Silhouette Score: {h1_score:.4f}"
            if h1_lo is not None and h1_hi is not None:
                h1_detail += f" | 95% CI: [{h1_lo:.4f}, {h1_hi:.4f}]"
            if h1_p is not None:
                h1_detail += f" | p-value: {_fmt_pval(h1_p)}"
            if not h1_detail:
                h1_detail = "—"
        else:
            h1_verdict = "Недостаточно данных"
            h1_detail = "—"

        # ── H2: Specialty Differentiation (chi-squared, cross-specialty) ──
        if h2_data:
            h2_p = h2_data.get("p_value")
            h2_cv = h2_data.get("cramers_v")
            h2_verdict = _verdict_from_pval(h2_p)
            h2_detail = f"p-value: {_fmt_pval(h2_p)}"
            if h2_cv is not None:
                h2_detail += f" | Cramér&#39;s V: {h2_cv:.4f}"
        else:
            h2_verdict = "Недостаточно данных (требуется ≥2 специальности)"
            h2_detail = "—"

        # ── H3: Experience Stratification ──
        if h3_data:
            h3_p = h3_data.get("p_value")
            h3_cv = h3_data.get("cramers_v")
            h3_anova = h3_data.get("anova") or {}
            h3_f = h3_anova.get("f_statistic")
            h3_verdict = _verdict_from_pval(h3_p)
            h3_levels = h3_data.get("experience_levels", [])
            h3_detail = f"p-value: {_fmt_pval(h3_p)}"
            if h3_cv is not None:
                h3_detail += f" | Cramér&#39;s V: {h3_cv:.4f}"
            if h3_f is not None:
                h3_detail += f" | F-статистика: {h3_f:.4f}"
        else:
            h3_verdict = "Недостаточно данных"
            h3_detail = "—"
            h3_p = None
            h3_cv = None
            h3_levels = []

        # ── H4: Curriculum-Gap Bootstrap ──
        if h4_data:
            h4_gap = h4_data.get("mean_jaccard")
            h4_lo = h4_data.get("ci_low")
            h4_hi = h4_data.get("ci_high")
            h4_is_sig = h4_data.get("is_significant")
            h4_verdict = (
                "Значимый разрыв (95% CI не включает 1.0)"
                if h4_is_sig
                else "Значимый разрыв (95% CI не включает 1.0)" if h4_is_sig is True
                else "Разрыв незначим (95% CI включает 1.0)"
            )
            if h4_is_sig is None:
                h4_verdict = "Недостаточно данных"
            h4_detail = ""
            if h4_gap is not None:
                h4_detail += f"Jaccard: {h4_gap:.4f}"
            if h4_lo is not None and h4_hi is not None:
                h4_detail += f" | 95% CI: [{h4_lo:.4f}, {h4_hi:.4f}]"
            if not h4_detail:
                h4_detail = "—"
        else:
            h4_verdict = "Недостаточно данных"
            h4_detail = "—"

        # ── H5: Taxonomy Stability (bootstrap ARI) ──
        if h5_data:
            h5_ari = h5_data.get("rand_index_mean")
            h5_lo = h5_data.get("rand_index_ci_low")
            h5_hi = h5_data.get("rand_index_ci_high")
            if h5_ari is not None:
                if h5_ari >= 0.7:
                    h5_verdict = f"Высокая стабильность (ARI={h5_ari:.3f})"
                elif h5_ari >= 0.4:
                    h5_verdict = f"Умеренная стабильность (ARI={h5_ari:.3f})"
                else:
                    h5_verdict = f"Низкая стабильность (ARI={h5_ari:.3f})"
            else:
                h5_verdict = "Недостаточно данных"
            h5_detail = f"ARI среднее: {h5_ari:.4f}" if h5_ari is not None else "—"
            if h5_lo is not None and h5_hi is not None:
                h5_detail += f" | 95% CI: [{h5_lo:.4f}, {h5_hi:.4f}]"
        else:
            h5_ari = None
            h5_verdict = "Недостаточно данных"
            h5_detail = "—"

        stats_section = f"""
<!-- Section: Statistical Validation (H1-H5) -->
<div id="stats">
<h2>📈 Статистическая валидация (H1-H5)</h2>
<details><summary><strong>H1: Silhouette-тест</strong> — {h1_verdict}</summary><p>{h1_detail}</p></details>
<details><summary><strong>H2: Хи-квадрат + Cramér&#39;s V</strong> — {h2_verdict}</summary><p>{h2_detail}</p></details>
<details><summary><strong>H3: Опыт ANOVA</strong> — {h3_verdict}</summary><p>{h3_detail}</p><p>Интерпретация: {_h3_interpretation(h3_p, h3_cv or 0, h3_levels)}</p></details>
<details><summary><strong>H4: Curriculum gap (bootstrap)</strong> — {h4_verdict}</summary><p>{h4_detail}</p></details>
<details><summary><strong>H5: Стабильность таксономии (ARI)</strong> — {h5_verdict}</summary><p>{h5_detail}</p><p>Интерпретация: {_h5_interpretation(h5_ari or 0)}</p></details>
</div>
"""
    else:
        stats_section = '<div id="stats"><h2>📈 Статистическая валидация (H1-H5)</h2><p>Недостаточно данных</p></div>'

    html += stats_section

    # ── Section: Monte Carlo Simulation ──
    if mc_data:
        mc = mc_data.get("monte_carlo") or {}
        opt = mc_data.get("optimization") or {}
        pareto = mc_data.get("pareto_frontier") or []

        mc_rows = ""
        if mc and "mean_coverage" in mc:
            mc_rows += f"""<tr><td>Monte Carlo</td>
<td>{mc['mean_coverage']*100:.1f}%</td>
<td>{mc.get('ci_lower', 0)*100:.1f}%</td>
<td>{mc.get('ci_upper', 0)*100:.1f}%</td>
<td>{mc.get('skill_count', 0)}</td></tr>"""
        if opt and "final_coverage" in opt:
            mc_rows += f"""<tr><td>Greedy оптимизация</td>
<td>{opt['final_coverage']*100:.1f}%</td>
<td>—</td><td>—</td>
<td>{opt.get('skills_count', 0)}</td></tr>"""

        pareto_rows = ""
        for i, pt in enumerate(pareto[:5]):
            cov = pt.get("coverage", 0) if isinstance(pt, dict) else pt[0] if isinstance(pt, (list, tuple)) else 0
            cnt = pt.get("k", 0) if isinstance(pt, dict) else pt[1] if isinstance(pt, (list, tuple)) and len(pt) > 1 else 0
            if isinstance(cov, (int, float)):
                pareto_rows += f"<tr><td>{i+1}</td><td>{cov*100:.1f}%</td><td>{cnt}</td></tr>"

        mc_section = f"""
<!-- Section: Monte Carlo Simulation -->
<div id="monte-carlo">
<h2>🎲 Монте-Карло симуляция и оптимизация учебного плана</h2>
<h3>Результаты симуляции</h3>
<table>
  <thead><tr><th>Метод</th><th>Покрытие</th><th>CI нижн</th><th>CI верх</th><th>Навыков</th></tr></thead>
  <tbody>{mc_rows or '<tr><td colspan="5">Недостаточно данных</td></tr>'}</tbody>
</table>"""
        if pareto_rows:
            mc_section += f"""
<h3>Парето-фронт (топ-5)</h3>
<table>
  <thead><tr><th>#</th><th>Покрытие</th><th>Навыков</th></tr></thead>
  <tbody>{pareto_rows}</tbody>
</table>"""
        mc_section += "\n</div>"
    else:
        mc_section = '<div id="monte-carlo"><h2>🎲 Монте-Карло симуляция</h2><p>Симуляция не выполнялась</p></div>'

    html += mc_section

    html += "\n</body>\n</html>"

    html_path = os.path.join(output_dir, f"{specialty}_report_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def _generate_role_spider_chart(
    role_name: str, abstract_chars: list[dict]
) -> str:
    """Generate a spider/radar chart for a role's competency characteristics."""
    import plotly.graph_objects as go

    if not abstract_chars:
        return ""

    names = [c["name"] for c in abstract_chars]
    values = [c.get("coverage_pct", 0) for c in abstract_chars]
    # Close the polygon
    names_closed = names + [names[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=names_closed,
            fill="toself",
            name=role_name,
            line={"color": "#1f77b4", "width": 2},
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )

    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%"},
            "angularaxis": {"direction": "clockwise"},
        },
        height=400,
        margin={"l": 60, "r": 60, "t": 20, "b": 20},
        showlegend=False,
    )

    return _fig_to_html_iframe(fig)


def _fig_to_html_iframe(fig: Any) -> str:
    """Convert a Plotly figure to an HTML iframe string."""
    import plotly.io as pio

    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def _print_final_summary(
    results: dict[str, Any],
    output_dir: str,
    timestamp: str,
) -> None:
    """Print a consolidated pipeline summary table."""
    console.print("\n[bold blue]═══ Pipeline Summary ═══[/]\n")

    phases = results.get("phases", {})

    # Phase status table
    status_table = Table(title="Phase Execution")
    status_table.add_column("Phase", style="cyan")
    status_table.add_column("Status")
    status_table.add_column("Time", justify="right")

    for phase_key, phase_data in phases.items():
        status = phase_data.get("status", "unknown")
        elapsed = phase_data.get("elapsed_seconds", 0)
        status_icon = "[green]✓[/]" if status == "ok" else "[red]✗[/]"
        status_table.add_row(phase_key, status_icon, f"{elapsed:.1f}s")

    console.print(status_table)

    # Output files
    console.print("\n[bold]Output Files:[/]")
    output_tree = Tree(f"[cyan]{output_dir}/[/]")

    charts_dir = os.path.join(output_dir, "charts")
    if os.path.isdir(charts_dir):
        charts_node = output_tree.add("charts/")
        for f in sorted(os.listdir(charts_dir)):
            if f.endswith(".html"):
                charts_node.add(f)

    for f in sorted(os.listdir(output_dir)):
        if f.endswith((".html", ".xlsx", ".json")) and f != ".":
            output_tree.add(f)

    console.print(output_tree)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_specialty_names(raw: str, config: Any) -> list[str]:
    """Resolve 'all' or comma-separated specialty names from config."""
    if raw.strip().lower() == "all":
        return sorted(config.specialties.keys())
    return [s.strip() for s in raw.split(",") if s.strip()]


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize pipeline results for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _h3_interpretation(
    p_value: float | None,
    cramers_v: float,
    levels: list[str],
) -> str:
    """Generate Russian-language interpretation for H3 experience stratification.

    Translates chi-squared test results into a human-readable conclusion
    about whether skill profiles differ across experience levels.
    """
    if p_value is None:
        return (
            f"Недостаточно данных для оценки различий профилей навыков "
            f"по уровням опыта. Уровни: {', '.join(levels)}."
        )
    if p_value < 0.01:
        effect = (
            "сильный" if cramers_v > 0.3
            else "умеренный" if cramers_v > 0.1
            else "слабый"
        )
        return (
            f"Профили навыков значимо различаются по уровням опыта "
            f"(p < 0.01, Cramér's V = {cramers_v:.3f}, {effect} эффект). "
            f"Уровни: {', '.join(levels)}."
        )
    if p_value < 0.05:
        return (
            f"Профили навыков значимо различаются по уровням опыта "
            f"(p = {p_value:.3f}, Cramér's V = {cramers_v:.3f}). "
            f"Уровни: {', '.join(levels)}."
        )
    return (
        f"Профили навыков не различаются значимо по уровням опыта "
        f"(p = {p_value:.3f}). Уровни: {', '.join(levels)}."
    )


def _h5_interpretation(ari_mean: float) -> str:
    """Generate Russian-language interpretation for H5 taxonomy stability.

    Translates bootstrap ARI into a human-readable conclusion about
    the reproducibility of skill clusters across data samples.
    """
    if ari_mean >= 0.7:
        return (
            f"Высокая стабильность кластеров (ARI = {ari_mean:.3f}): "
            f"таксономия навыков устойчива к изменениям выборки."
        )
    if ari_mean >= 0.4:
        return (
            f"Умеренная стабильность кластеров (ARI = {ari_mean:.3f}): "
            f"таксономия частично воспроизводима."
        )
    return (
        f"Низкая стабильность кластеров (ARI = {ari_mean:.3f}): "
        f"таксономия неустойчива, требуется больше данных."
    )
