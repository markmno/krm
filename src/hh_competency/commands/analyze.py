"""Analyze commands: skill extraction, comparison, clustering, radar charts.

Orchestrates NLPPipeline → SkillAnalyzer → SkillComparator → SkillClusterer →
statistical testing → simulation, with Rich-formatted output.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from hh_competency.config import load_config
from hh_competency.storage.db import Database

if TYPE_CHECKING:
    from hh_competency.storage.models import AppConfig

console = Console()
analyze_app = typer.Typer(help="Analyze vacancy data: skills, roles, stats, simulation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_specialties(raw: str) -> list[str]:
    """Split a comma-separated specialty list into a cleaned list."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _resolve_specialties(
    raw: str | None, config: AppConfig, allow_all: bool = False
) -> list[str]:
    """Resolve specialty names from raw input or config defaults.

    If raw is None or 'all' and allow_all, returns all configured specialties.
    Otherwise splits the raw comma-separated string.
    """
    if raw is None:
        return sorted(config.specialties.keys()) if allow_all else []

    cleaned = _parse_specialties(raw)
    if allow_all and len(cleaned) == 1 and cleaned[0] == "all":
        return sorted(config.specialties.keys())

    return cleaned


def _validate_specialties(
    requested: list[str], config: AppConfig
) -> tuple[list[str], list[str]]:
    """Validate requested specialties against config, returning (valid, invalid)."""
    known = set(config.specialties.keys())
    valid = [s for s in requested if s in known]
    invalid = [s for s in requested if s not in known]
    return valid, invalid


def _build_db(config: AppConfig, db_path: str) -> Database:
    """Create a Database instance, falling back to config default path."""
    path = db_path or config.database_path
    return Database(path, read_only=True)


def _output_format_flag() -> Any:
    """Return a typer Option for output format selection."""
    return typer.Option("table", "--format", "-f", help="Output format: table, json, csv")


# ---------------------------------------------------------------------------
# analyze skills
# ---------------------------------------------------------------------------


@analyze_app.command(name="skills")
def analyze_skills(
    specialty: str = typer.Argument(..., help="Specialty name to analyze"),
    top: int = typer.Option(50, "--top", "-n", help="Number of top skills to show"),
    format: str = _output_format_flag(),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Extract and display top skills for a specialty.

    Uses NLPPipeline (pymorphy3) to lemmatize vacancy descriptions
    and rank skills by frequency.

    Example: hh-competency analyze skills physics --top 30 --format table
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        # Validate specialty
        _, invalid = _validate_specialties([specialty], app_config)
        if invalid:
            console.print(f"[red]Unknown specialty:[/] {invalid[0]}")
            available = ", ".join(sorted(app_config.specialties.keys()))
            console.print(f"[dim]Available: {available}[/]")
            raise typer.Exit(code=1)

        from hh_competency.analysis.skills import SkillAnalyzer
        from hh_competency.nlp.pipeline import NLPPipeline

        pipeline = NLPPipeline()
        analyzer = SkillAnalyzer(database, pipeline)
        skills = analyzer.top_skills(specialty, n=top)

        if not skills:
            console.print(f"[yellow]No skill data found for: {specialty}[/]")
            raise typer.Exit(code=0)

        if format == "json":
            data = [
                {
                    "lemma": s.lemma,
                    "pos": s.pos,
                    "frequency": s.frequency,
                    "vacancy_count": s.vacancy_count,
                }
                for s in skills
            ]
            console.print(json.dumps(data, ensure_ascii=False, indent=2))
        elif format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["lemma", "pos", "frequency", "vacancy_count"])
            for s in skills:
                writer.writerow([s.lemma, s.pos, s.frequency, s.vacancy_count])
            console.print(output.getvalue())
        else:
            table = Table(title=f"Top {len(skills)} Skills: {specialty}")
            table.add_column("Rank", style="dim", justify="right")
            table.add_column("Skill", style="cyan")
            table.add_column("POS", style="dim")
            table.add_column("Frequency", justify="right", style="green")
            table.add_column("Vacancy Count", justify="right", style="yellow")

            for i, s in enumerate(skills, 1):
                table.add_row(str(i), s.lemma, s.pos, str(s.frequency), str(s.vacancy_count))

            console.print(table)

    finally:
        database.close()


# ---------------------------------------------------------------------------
# analyze compare
# ---------------------------------------------------------------------------


@analyze_app.command(name="compare")
def analyze_compare(
    specialties: str = typer.Argument(
        ..., help="Comma-separated specialties to compare (e.g. physics,biology)"
    ),
    top: int = typer.Option(30, "--top", "-n", help="Number of top skills per specialty"),
    format: str = _output_format_flag(),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Compare skill profiles across specialties.

    Shows top skills side-by-side with frequencies for cross-specialty analysis.

    Example: hh-competency analyze compare physics,biology --top 20
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        requested = _parse_specialties(specialties)
        valid, invalid = _validate_specialties(requested, app_config)
        if invalid:
            console.print(f"[red]Unknown specialties:[/] {', '.join(invalid)}")
        if not valid:
            console.print("[red]No valid specialties specified[/]")
            raise typer.Exit(code=1)

        from hh_competency.analysis.compare import SkillComparator

        comparator = SkillComparator(database)
        result = comparator.compare(valid, top_n=top)

        if format == "json":
            data = {}
            for spec, skills in result.items():
                data[spec] = [
                    {"lemma": s.lemma, "frequency": s.frequency} for s in skills
                ]
            console.print(json.dumps(data, ensure_ascii=False, indent=2))
        elif format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["specialty", "lemma", "frequency"])
            for spec in sorted(result):
                for s in result[spec]:
                    writer.writerow([spec, s.lemma, s.frequency])
            console.print(output.getvalue())
        else:
            # Build unified table
            all_lemmas: dict[str, dict[str, int]] = {}
            for spec, skills in result.items():
                for s in skills:
                    if s.lemma not in all_lemmas:
                        all_lemmas[s.lemma] = {}
                    all_lemmas[s.lemma][spec] = s.frequency

            # Sort by max frequency across all specialties
            sorted_lemmas = sorted(
                all_lemmas.items(),
                key=lambda item: max(item[1].values()),
                reverse=True,
            )

            table = Table(title=f"Skill Comparison ({', '.join(valid)})")
            table.add_column("Skill", style="cyan")
            for spec in valid:
                table.add_column(spec, justify="right", style="green")

            for lemma, freqs in sorted_lemmas:
                row = [lemma]
                for spec in valid:
                    row.append(str(freqs.get(spec, 0)))
                table.add_row(*row)

            console.print(table)

    finally:
        database.close()


# ---------------------------------------------------------------------------
# analyze radar
# ---------------------------------------------------------------------------


@analyze_app.command(name="radar")
def analyze_radar(
    specialties: str = typer.Argument(
        ..., help="Comma-separated specialties (min 2, e.g. physics,biology,chemistry)"
    ),
    top: int = typer.Option(8, "--top", "-n", help="Number of differentiating skill axes"),
    output: str = typer.Option(
        "radar.html", "--output", "-o", help="Output HTML file path"
    ),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Generate a spider/radar chart comparing skill profiles across specialties.

    Axes are the top N most differentiating skills across all specialty pairs.
    Output is a self-contained HTML file with Plotly chart.

    Example: hh-competency analyze radar physics,biology,chemistry --top 10
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        requested = _parse_specialties(specialties)
        valid, invalid = _validate_specialties(requested, app_config)
        if invalid:
            console.print(f"[red]Unknown specialties:[/] {', '.join(invalid)}")
        if len(valid) < 2:
            console.print("[red]Need at least 2 valid specialties for radar chart[/]")
            raise typer.Exit(code=1)

        from hh_competency.viz.radar import generate_radar_chart

        console.print(f"[bold blue]Generating radar chart:[/] {', '.join(valid)}")
        generate_radar_chart(database, valid, top_n_skills=top, output_path=output)
        console.print(f"[green]Radar chart saved to:[/] {os.path.abspath(output)}")
        console.print(
            f"  [dim]Specialties: {len(valid)}, "
            f"Skill axes: {top}, "
            f"Chart type: radar[/]"
        )

    finally:
        database.close()


# ---------------------------------------------------------------------------
# analyze clusters
# ---------------------------------------------------------------------------


@analyze_app.command(name="clusters")
def analyze_clusters(
    specialty: str = typer.Argument(..., help="Specialty name to analyze"),
    clusters: int = typer.Option(3, "--clusters", "-k", help="Target number of role clusters"),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Discover competency-role profiles via skill co-occurrence clustering.

    Uses AgglomerativeClustering with Jaccard distance to group skills into
    role profiles (e.g. "python backend разработчик").

    Example: hh-competency analyze clusters physics --clusters 4
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        _, invalid = _validate_specialties([specialty], app_config)
        if invalid:
            console.print(f"[red]Unknown specialty:[/] {invalid[0]}")
            raise typer.Exit(code=1)

        from hh_competency.analysis.clustering import SkillClusterer

        console.print(
            f"[bold blue]Discovering roles for:[/] {specialty} "
            f"(target: {clusters} clusters)"
        )

        clusterer = SkillClusterer(database)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[cyan]Clustering skills...", total=None
            )
            roles = clusterer.discover_roles(specialty, n_clusters=clusters)
            progress.update(task_id, completed=100, visible=False)

        if not roles:
            console.print(f"[yellow]No role profiles discovered for: {specialty}[/]")
            raise typer.Exit(code=0)

        console.print(
            f"\n[bold green]Discovered {len(roles)} role profiles[/]\n"
        )

        for role in roles:
            panel_content = ""
            if role.defining_skills:
                panel_content += "[bold]Определяющие навыки:[/]\n"
                for lemma, weight in role.defining_skills:
                    panel_content += f"  • [cyan]{lemma}[/] (вес: {weight:.1f})\n"
            if role.supporting_skills:
                panel_content += "[bold dim]Поддерживающие навыки:[/]\n"
                for lemma, weight in role.supporting_skills:
                    panel_content += f"  • [dim]{lemma}[/] (вес: {weight:.1f})\n"

            console.print(
                Panel(panel_content.strip(), title=f"[bold]{role.role_name}[/]")
            )

    finally:
        database.close()


# ---------------------------------------------------------------------------
# analyze roles
# ---------------------------------------------------------------------------


@analyze_app.command(name="roles")
def analyze_roles(
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Discover competency-role profiles for ALL specialties in config.

    Runs SkillClusterer for each specialty and shows discovered roles
    with defining and supporting skills.

    Example: hh-competency analyze roles
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        specialties = sorted(app_config.specialties.keys())
        console.print(
            f"[bold blue]Analyzing roles across {len(specialties)} specialties[/]\n"
        )

        from hh_competency.analysis.clustering import SkillClusterer

        clusterer = SkillClusterer(database)
        total_roles = 0

        for spec in specialties:
            roles = clusterer.discover_roles(spec)
            if not roles:
                console.print(f"  [dim]{spec}: no roles discovered[/]")
                continue

            console.print(f"\n[bold]{spec}[/] — {len(roles)} role(s)")
            for role in roles:
                skills_str = ", ".join(
                    lemma for lemma, _w in role.defining_skills[:5]
                )
                console.print(f"  [cyan]{role.role_name}[/]: {skills_str}")
            total_roles += len(roles)

        console.print(f"\n[green]Total roles discovered: {total_roles}[/]")

    finally:
        database.close()


# ---------------------------------------------------------------------------
# analyze stats
# ---------------------------------------------------------------------------


@analyze_app.command(name="stats")
def analyze_stats(
    specialty: str = typer.Argument(..., help="Specialty name to analyze"),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Run statistical tests for a specialty's role clusters.

    Tests cluster significance (bootstrap silhouette), cluster stability
    (bootstrap ARI, chi-squared differentiation), and optimal cluster count.

    Example: hh-competency analyze stats physics
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        _, invalid = _validate_specialties([specialty], app_config)
        if invalid:
            console.print(f"[red]Unknown specialty:[/] {invalid[0]}")
            raise typer.Exit(code=1)

        from hh_competency.stats.significance import ClusterSignificanceTester
        from hh_competency.stats.stability import ClusterStabilityTester

        console.print(f"[bold blue]Statistical analysis for:[/] {specialty}\n")

        # 1. Cluster significance test
        console.print("[bold]1. Bootstrap Cluster Significance[/]")
        sig_tester = ClusterSignificanceTester(database)
        sig_result = sig_tester.test_cluster_significance(specialty, n_iter=100)

        sig_content = (
            f"Silhouette score: [cyan]{sig_result['silhouette_score']}[/]\n"
            f"95% CI: [{sig_result['silhouette_ci_low']}, {sig_result['silhouette_ci_high']}]\n"
            f"p-value: "
        )
        if sig_result["p_value"] < 0.01:
            sig_content += "[bold green]p < 0.01[/] (highly significant)"
        elif sig_result["p_value"] < 0.05:
            sig_content += "[green]p < 0.05[/] (significant)"
        else:
            sig_content += f"[yellow]p = {sig_result['p_value']}[/] (not significant)"

        console.print(Panel(sig_content.strip(), title="Significance Test"))

        # 2. Optimal cluster count
        console.print("\n[bold]2. Optimal Cluster Count[/]")
        cluster_results = sig_tester.test_cluster_count(specialty, max_k=6)

        if cluster_results:
            table = Table(title="Cluster Count Evaluation")
            table.add_column("k", justify="right", style="cyan")
            table.add_column("Silhouette", justify="right", style="green")
            table.add_column("Gap Statistic", justify="right", style="yellow")
            table.add_column("Optimal", justify="center")

            for r in cluster_results:
                gap_str = f"{r['gap_statistic']:.4f}" if r["gap_statistic"] is not None else "N/A"
                optimal_mark = "✓" if r.get("is_optimal") else ""
                table.add_row(
                    str(r["k"]),
                    str(r["silhouette_score"]),
                    gap_str,
                    optimal_mark,
                )

            console.print(table)
        else:
            console.print("[dim]Insufficient data for cluster count analysis[/]")

        # 3. Chi-squared differentiation (across all specialties)
        console.print("\n[bold]3. Specialty Differentiation (Chi-squared)[/]")
        stab_tester = ClusterStabilityTester(database)
        all_specs = sorted(app_config.specialties.keys())
        chi2_result = stab_tester.test_specialty_differentiation(all_specs)

        chi2_content = (
            f"χ² = [cyan]{chi2_result['chi2_statistic']}[/], "
            f"df = {chi2_result['dof']}, "
        )
        if chi2_result["p_value"] < 0.001:
            chi2_content += "[bold green]p < 0.001[/] (strongly significant)"
        elif chi2_result["p_value"] < 0.05:
            chi2_content += f"[green]p = {chi2_result['p_value']:.4f}[/] (significant)"
        else:
            chi2_content += f"[yellow]p = {chi2_result['p_value']:.4f}[/]"
        chi2_content += f"\nCramér's V = [cyan]{chi2_result['cramers_v']:.4f}[/] (effect size)"

        console.print(Panel(chi2_content.strip(), title="Chi-squared Test"))

        # 4. Bootstrap ARI (stability)
        console.print("\n[bold]4. Bootstrap Cluster Stability (ARI)[/]")
        rand_result = stab_tester.bootstrap_rand_index(
            specialty, n_iter=50, n_clusters=3
        )

        if rand_result["n_iter"] > 0:
            ari_content = (
                f"Mean ARI: [cyan]{rand_result['rand_index_mean']:.4f}[/]\n"
                f"95% CI: [{rand_result['rand_index_ci_low']:.4f}, "
                f"{rand_result['rand_index_ci_high']:.4f}]\n"
                f"Iterations: {rand_result['n_iter']}"
            )
        else:
            ari_content = "[yellow]Insufficient data for bootstrap stability[/]"

        console.print(Panel(ari_content.strip(), title="Bootstrap ARI"))

    finally:
        database.close()


# ---------------------------------------------------------------------------
# analyze simulate
# ---------------------------------------------------------------------------


@analyze_app.command(name="simulate")
def analyze_simulate(
    specialty: str = typer.Argument(..., help="Specialty name to simulate"),
    iterations: int = typer.Option(
        1000, "--iterations", "-n", help="Number of Monte Carlo iterations"
    ),
    skills: str = typer.Option(
        "",
        "--skills",
        "-s",
        help="Comma-separated curriculum skills (default: auto-select top skills)",
    ),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Run Monte Carlo simulation to estimate curriculum-vacancy fit.

    Bootstraps vacancy samples to estimate the distribution of Jaccard fit
    between curriculum skills and market demands, producing confidence intervals.

    Example: hh-competency analyze simulate physics --iterations 500 --skills "python,sql,git"
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        _, invalid = _validate_specialties([specialty], app_config)
        if invalid:
            console.print(f"[red]Unknown specialty:[/] {invalid[0]}")
            raise typer.Exit(code=1)

        from hh_competency.analysis.skills import SkillAnalyzer
        from hh_competency.nlp.pipeline import NLPPipeline
        from hh_competency.simulation.monte_carlo import MonteCarloEngine

        pipeline = NLPPipeline()

        # Determine curriculum skills
        if skills.strip():
            curriculum_skills = _parse_specialties(skills)
            console.print(
                f"[dim]Using {len(curriculum_skills)} specified skills[/]"
            )
        else:
            # Auto-select top skills from DB
            analyzer = SkillAnalyzer(database, pipeline)
            top_skills = analyzer.top_skills(specialty, n=20)
            curriculum_skills = [s.lemma for s in top_skills]
            console.print(
                f"[dim]Auto-selected top {len(curriculum_skills)} skills[/]"
            )

        if not curriculum_skills:
            console.print("[red]No curriculum skills available[/]")
            raise typer.Exit(code=1)

        console.print(
            f"[bold blue]Monte Carlo simulation for:[/] {specialty}\n"
            f"  Skills: {len(curriculum_skills)}\n"
            f"  Iterations: {iterations}\n"
        )

        engine = MonteCarloEngine(database, pipeline)
        result = engine.simulate_curriculum_fit(
            curriculum_skills=curriculum_skills,
            specialty=specialty,
            n_iter=iterations,
        )

        # Format output
        table = Table(title="Simulation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Scenario", result.scenario_name)
        table.add_row("Specialty", result.specialty)
        table.add_row("Iterations", str(result.n_samples))
        table.add_row("Skills tested", str(len(result.skill_set)))
        table.add_row(
            "Mean coverage",
            f"{result.mean_coverage:.4f} ({result.mean_coverage*100:.1f}%)",
        )
        table.add_row("95% CI lower", f"{result.ci_lower:.4f} ({result.ci_lower*100:.1f}%)")
        table.add_row("95% CI upper", f"{result.ci_upper:.4f} ({result.ci_upper*100:.1f}%)")

        console.print(table)

        # Interpretation
        ci_width = result.ci_upper - result.ci_lower
        console.print(
            f"\n[bold]Interpretation:[/] "
            f"Curriculum covers [cyan]{result.mean_coverage*100:.1f}%[/] "
            f"of market vacancies on average "
            f"(95% CI: [{result.ci_lower*100:.1f}% – {result.ci_upper*100:.1f}%], "
            f"width: {ci_width*100:.1f}pp)"
        )

    finally:
        database.close()


@analyze_app.command(name="trends")
def analyze_trends(
    specialty: str = typer.Argument(..., help="Specialty name to analyze trends"),
    db_path: str = typer.Option("data/hh_competency.db", "--db"),
    top_n: int = typer.Option(30, "--top", "-n", help="Top N skills to trend"),
    min_count: int = typer.Option(2, "--min-count", help="Min frequency to include"),
) -> None:
    """Yearly trend analysis — emerging/declining/stable skills and salary trends."""
    from rich.table import Table

    from hh_competency.analysis.trends import analyze_salary_trends
    from hh_competency.analysis.trends import analyze_trends as _atrends
    from hh_competency.nlp.pipeline import NLPPipeline
    from hh_competency.storage.db import Database

    pipeline = NLPPipeline()
    db = Database(db_path, read_only=True)
    try:
        r = _atrends(db, pipeline, specialty, top_n=top_n, min_count=min_count)
        if "error" in r:
            console.print(f"[red]{r['error']}[/]")
            return

        console.print(f"\n[bold cyan]Yearly Trends: {specialty}[/]")
        console.print(f"Years: {r.get('years_analyzed', [])} | Skills: {r.get('total_skills', 0)}")

        s, c = r.get("sustainable", False), "green" if r.get("sustainable", False) else "yellow"
        console.print(f"\n[bold {c}]Verdict: {r.get('verdict', '?')}[/]")

        for label, style, data in [
            ("Emerging (p < 0.05)", "green", r.get("emerging_skills", [])),
            ("Declining (p < 0.05)", "red", r.get("declining_skills", [])),
        ]:
            if data:
                console.print(f"\n[bold {style}]{label}:[/]")
                t = Table(show_header=True, header_style=style)
                for h in ["Skill", "Slope", "p", "Tau"]:
                    t.add_column(h)
                for s in data:
                    t.add_row(
                        s["skill"],
                        f"{s.get('linear_slope', 0):+.2f}",
                        f"{s.get('p_value', 0):.3f}",
                        f"{s.get('tau', 0):.2f}",
                    )
                console.print(t)

        if r.get("stable_count", 0):
            console.print(f"\n[dim]Stable: {r['stable_count']}[/]")

        sal = analyze_salary_trends(db, specialty)
        if "error" not in sal:
            st = sal.get("trend", {})
            console.print("\n[bold cyan]Salary Trends[/]")
            console.print(f"Median: {sal.get('median_slope', 0):+.0f} RUB/yr")
            console.print(f"Direction: {st.get('direction', '?')} (p={st.get('p_value', 1):.4f})")

    finally:
        db.close()
