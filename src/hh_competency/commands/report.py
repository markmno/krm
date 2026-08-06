"""Report commands: HTML report generation, data export, cross-specialty summary.

Produces self-contained HTML reports with embedded Plotly charts, Excel/CSV exports,
and Rich-formatted terminal summaries covering all analysis dimensions.
"""

from __future__ import annotations

import base64
import csv
import os
from datetime import UTC, datetime
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from hh_competency.config import load_config
from hh_competency.storage.db import Database
from hh_competency.storage.models import AppConfig

console = Console()
report_app = typer.Typer(help="Generate reports, export data, view summaries")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_specialties(raw: str) -> list[str]:
    """Split a comma-separated specialty list into a cleaned list."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _resolve_specialty_names(raw: str, config: AppConfig) -> list[str]:
    """Resolve 'all' or comma-separated specialty names from config."""
    if raw.strip().lower() == "all":
        return sorted(config.specialties.keys())
    return _parse_specialties(raw)


def _build_db(config: AppConfig, db_path: str) -> Database:
    """Create a read-only Database instance."""
    path = db_path or config.database_path
    return Database(path, read_only=True)


def _fig_to_base64(fig: Any) -> str:
    """Convert a Plotly figure to a base64-encoded PNG for HTML embedding.

    Tries kaleido (plotly.io.to_image) first for a static PNG;
    falls back to an HTML iframe if kaleido is unavailable.
    """
    try:
        import plotly.io as pio

        img_bytes = pio.to_image(fig, format="png", width=1000, height=600, scale=2)
        return base64.b64encode(img_bytes).decode("ascii")
    except Exception:
        # Fallback: embed as iframe
        return ""


def _fig_to_html_iframe(fig: Any) -> str:
    """Convert a Plotly figure to an HTML iframe string."""
    import plotly.io as pio

    html_str = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    return html_str


def _timestamp() -> str:
    """Return current UTC timestamp as ISO format string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")


# ---------------------------------------------------------------------------
# report generate
# ---------------------------------------------------------------------------


@report_app.command(name="generate")
def report_generate(
    specialty: str = typer.Argument(..., help="Specialty name for report"),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
    format: str = typer.Option("html", "--format", "-f", help="Report format: html"),
    output_dir: str = typer.Option(
        "reports/", "--output-dir", "-o", help="Output directory for reports"
    ),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
) -> None:
    """Generate a full HTML report with embedded charts for a specialty.

    Includes: summary stats, top skills bar chart, role profiles,
    salary boxplot, and FGOS curriculum comparison.

    Example: hh-competency report generate physics --format html
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        if specialty not in app_config.specialties:
            console.print(f"[red]Unknown specialty:[/] {specialty}")
            available = ", ".join(sorted(app_config.specialties.keys()))
            console.print(f"[dim]Available: {available}[/]")
            raise typer.Exit(code=1)

        ts = _timestamp()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{specialty}_report_{ts}.html")

        console.print(f"[bold blue]Generating report for:[/] {specialty}")

        # --- Collect data ---
        vacancy_count = database.get_vacancy_count(specialty)
        top_skills = database.get_skill_frequencies(specialty, top_n=30)
        salary_data = database.get_salary_data(specialty)

        # --- Generate charts ---
        from hh_competency.viz.bars import (
            generate_role_profile_chart,
            generate_skill_bar_chart,
        )
        from hh_competency.viz.salary import generate_salary_boxplot

        console.print("  [dim]Generating skill bar chart...[/]")
        skills_fig = generate_skill_bar_chart(database, specialty, top_n=20)
        skills_img = _fig_to_base64(skills_fig)
        skills_html = _fig_to_html_iframe(skills_fig) if not skills_img else ""

        console.print("  [dim]Generating role profile chart...[/]")
        try:
            roles_fig = generate_role_profile_chart(database, specialty)
            roles_img = _fig_to_base64(roles_fig)
            roles_html = _fig_to_html_iframe(roles_fig) if not roles_img else ""
        except Exception:
            roles_img = ""
            roles_html = (
                "<p style='color: #888; font-style: italic;'>"
                "Недостаточно данных для построения ролевых профилей</p>"
            )

        console.print("  [dim]Generating salary boxplot...[/]")
        try:
            salary_fig = generate_salary_boxplot(database, [specialty])
            salary_img = _fig_to_base64(salary_fig)
            salary_html = _fig_to_html_iframe(salary_fig) if not salary_img else ""
            has_salary = True
        except Exception:
            has_salary = False
            salary_img = ""
            salary_html = (
                "<p style='color: #888; font-style: italic;'>"
                "Нет данных о зарплатах для этой специальности</p>"
            )

        # --- FGOS comparison (conditional import) ---
        fgos_html = ""
        fgos_data: dict[str, Any] = {}
        try:
            from hh_competency.methodology.fgos import FgosComparator

            fgos_comparator = FgosComparator(database)
            fgos_data = fgos_comparator.compare_specialty(specialty)
        except ImportError:
            fgos_html = (
                "<p style='color: #888; font-style: italic;'>"
                "Модуль сравнения с ФГОС не загружен</p>"
            )
        except Exception:
            fgos_html = (
                "<p style='color: #888; font-style: italic;'>"
                "Ошибка при сравнении с ФГОС</p>"
            )

        # --- Profstandart mapping (conditional import) ---
        profstandart_html = ""
        try:
            from hh_competency.methodology.profstandart import ProfstandartMapper

            ps_mapper = ProfstandartMapper(database)
            ps_data = ps_mapper.map_roles_to_profstandart(specialty)
            if ps_data:
                ps_rows = "".join(
                    f"<tr><td>{r.get('role', '')}</td>"
                    f"<td>{r.get('profstandart', '')}</td></tr>"
                    for r in ps_data
                )
                profstandart_html = (
                    "<h2>Соответствие профстандартам</h2>\n"
                    "<table><thead><tr><th>Роль</th><th>Профстандарт</th></tr></thead>\n"
                    f"<tbody>{ps_rows}</tbody></table>"
                )
        except ImportError:
            profstandart_html = (
                "<p style='color: #888; font-style: italic;'>"
                "Модуль сопоставления с профстандартами не загружен</p>"
            )
        except Exception:
            profstandart_html = (
                "<p style='color: #888; font-style: italic;'>"
                "Ошибка при сопоставлении с профстандартами</p>"
            )

        # --- Build HTML ---
        skills_section = ""
        if skills_img:
            skills_section = (
                "<h2>Топ навыков</h2>\n"
                f'<img src="data:image/png;base64,{skills_img}" '
                'style="max-width:100%; height:auto;" alt="Skills chart">'
            )
        elif skills_html:
            skills_section = f"<h2>Топ навыков</h2>\n{skills_html}"

        roles_section = ""
        if roles_img:
            roles_section = (
                "<h2>Профили ролей</h2>\n"
                f'<img src="data:image/png;base64,{roles_img}" '
                'style="max-width:100%; height:auto;" alt="Role profiles">'
            )
        elif roles_html or (isinstance(roles_html, str) and roles_html):
            roles_section = f"<h2>Профили ролей</h2>\n{roles_html}"

        salary_section = ""
        if salary_img:
            salary_section = (
                "<h2>Заработная плата</h2>\n"
                f'<img src="data:image/png;base64,{salary_img}" '
                'style="max-width:100%; height:auto;" alt="Salary boxplot">'
            )
        elif salary_html or (isinstance(salary_html, str) and salary_html):
            salary_section = f"<h2>Заработная плата</h2>\n{salary_html}"

        # Skills stats table
        skills_rows = ""
        for s in top_skills[:20]:
            skills_rows += (
                f"<tr><td>{s.lemma}</td>"
                f"<td>{s.frequency}</td>"
                f"<td>{s.vacancy_count}</td></tr>\n"
            )

        # Salary stats table
        salary_rows = ""
        if has_salary and not salary_data.empty:
            try:
                from hh_competency.viz.salary import _compute_salary_avg

                avg_sal = _compute_salary_avg(salary_data)
                avg_sal = avg_sal[avg_sal > 0]
                if not avg_sal.empty:
                    med = avg_sal.median()
                    q1 = avg_sal.quantile(0.25)
                    q3 = avg_sal.quantile(0.75)
                    salary_rows = (
                        f"<tr><td>Медиана</td><td>{med:,.0f} ₽</td></tr>\n"
                        f"<tr><td>Q1 (25%)</td><td>{q1:,.0f} ₽</td></tr>\n"
                        f"<tr><td>Q3 (75%)</td><td>{q3:,.0f} ₽</td></tr>\n"
                        f"<tr><td>Вакансий с зарплатой</td><td>{len(avg_sal)}</td></tr>\n"
                    )
            except Exception:
                salary_rows = ""

        fgos_section = ""
        if fgos_data:
            fgos_section = "<h2>Соответствие ФГОС</h2>\n"
            if "recommendations" in fgos_data:
                recs = fgos_data["recommendations"]
                if isinstance(recs, list):
                    fgos_section += "<ul>\n" + "".join(
                        f"<li>{r}</li>\n" for r in recs
                    ) + "</ul>\n"
                else:
                    fgos_section += f"<p>{recs}</p>\n"
        elif fgos_html:
            fgos_section = f"<h2>Соответствие ФГОС</h2>\n{fgos_html}"

        recommendations_html = ""
        if not fgos_html and not fgos_data:
            recommendations_html = (
                "<h2>Рекомендации</h2>\n"
                "<p>Для получения рекомендаций запустите модуль анализа ФГОС</p>"
            )

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Отчёт: {specialty}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; background: #fff; }}
  h1 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; margin-top: 40px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #1f77b4; color: #fff; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
  img {{ max-width: 100%; height: auto; margin: 16px 0; }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }}
  .stat-card {{ background: #f0f4f8; padding: 16px; border-radius: 8px; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: bold; color: #1f77b4; }}
  .stat-card .label {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Отчёт по специальности: {specialty}</h1>
<div class="meta">
  Сгенерировано: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
  Вакансий в базе: {vacancy_count}<br>
</div>

<h2>Статистика</h2>
<div class="stats-grid">
  <div class="stat-card">
    <span class="value">{vacancy_count}</span><br>
    <span class="label">Всего вакансий</span>
  </div>
  <div class="stat-card">
    <span class="value">{len(top_skills)}</span><br>
    <span class="label">Уникальных навыков</span>
  </div>
</div>

<h2>Топ-20 навыков</h2>
<table>
  <thead><tr><th>Навык</th><th>Частота</th><th>Вакансий</th></tr></thead>
  <tbody>{skills_rows}</tbody>
</table>

{skills_section}

{roles_section}

{salary_section}
<table>
  <thead><tr><th>Метрика</th><th>Значение</th></tr></thead>
  <tbody>{salary_rows}</tbody>
</table>

{fgos_section}

{recommendations_html}

{profstandart_html}

</body>
</html>"""

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        console.print(f"[green]Report saved to:[/] {os.path.abspath(output_path)}")

    finally:
        database.close()


# ---------------------------------------------------------------------------
# report export
# ---------------------------------------------------------------------------


@report_app.command(name="export")
def report_export(
    format: str = typer.Option(
        "excel", "--format", "-f", help="Export format: excel, csv"
    ),
    specialties: str = typer.Option(
        "all",
        "--specialties",
        "-s",
        help="Specialties to export: 'all' or comma-separated names",
    ),
    output: str = typer.Option(
        "", "--output", "-o", help="Output file path (auto-generated if empty)"
    ),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Export analyzed data to Excel (.xlsx) or CSV format.

    Excel export includes multiple sheets: Overview, Skills, Roles, Salaries, Recommendations.
    CSV export includes one file per specialty.

    Example: hh-competency report export --format excel --specialties physics,biology
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        spec_names = _resolve_specialty_names(specialties, app_config)
        if not spec_names:
            console.print("[red]No valid specialties to export[/]")
            raise typer.Exit(code=1)

        ts = _timestamp()

        if format == "csv":
            for spec in spec_names:
                csv_path = output or f"export_{spec}_{ts}.csv"
                skills = database.get_skill_frequencies(spec, top_n=200)
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["specialty", "lemma", "pos", "frequency", "vacancy_count"])
                    for s in skills:
                        writer.writerow([s.specialty, s.lemma, s.pos, s.frequency, s.vacancy_count])
                console.print(f"[green]CSV saved:[/] {os.path.abspath(csv_path)}")

        elif format == "excel":
            xlsx_path = output or f"export_{ts}.xlsx"

            try:
                import openpyxl
            except ImportError:
                console.print(
                    "[red]openpyxl not installed. Run: uv add openpyxl[/]"
                )
                raise typer.Exit(code=1) from None

            wb = openpyxl.Workbook()

            # --- Sheet 1: Обзор ---
            ws_overview = wb.active
            ws_overview.title = "Обзор"
            ws_overview.append(["Специальность", "Вакансий", "Навыков", "Ролей"])
            for spec in spec_names:
                vc = database.get_vacancy_count(spec)
                sc = len(database.get_skill_frequencies(spec, top_n=500))
                try:
                    from hh_competency.analysis.clustering import SkillClusterer

                    clusterer = SkillClusterer(database)
                    roles = clusterer.discover_roles(spec)
                    rc = len(roles)
                except Exception:
                    rc = 0
                ws_overview.append([spec, vc, sc, rc])

            # --- Sheet 2: Навыки ---
            ws_skills = wb.create_sheet("Навыки")
            ws_skills.append(["Специальность", "Навык", "POS", "Частота", "Вакансий"])
            for spec in spec_names:
                skills = database.get_skill_frequencies(spec, top_n=200)
                for s in skills:
                    ws_skills.append([s.specialty, s.lemma, s.pos, s.frequency, s.vacancy_count])

            # --- Sheet 3: Роли ---
            ws_roles = wb.create_sheet("Роли")
            ws_roles.append([
                "Специальность", "Роль", "Определяющие навыки", "Поддерживающие навыки"
            ])
            for spec in spec_names:
                try:
                    from hh_competency.analysis.clustering import SkillClusterer

                    clusterer = SkillClusterer(database)
                    roles = clusterer.discover_roles(spec)
                    for role in roles:
                        defining = ", ".join(lemma for lemma, _w in role.defining_skills)
                        supporting = ", ".join(lemma for lemma, _w in role.supporting_skills)
                        ws_roles.append([spec, role.role_name, defining, supporting])
                except Exception:
                    pass

            # --- Sheet 4: Зарплаты ---
            ws_salary = wb.create_sheet("Зарплаты")
            ws_salary.append(["Специальность", "От", "До", "Валюта", "Опыт", "Город"])
            for spec in spec_names:
                try:
                    sal_df = database.get_salary_data(spec)
                    for _, row in sal_df.iterrows():
                        ws_salary.append([
                            spec,
                            row.get("salary_from", ""),
                            row.get("salary_to", ""),
                            row.get("currency", ""),
                            row.get("experience", ""),
                            row.get("area", ""),
                        ])
                except Exception:
                    pass

            # --- Sheet 5: Рекомендации ---
            ws_recs = wb.create_sheet("Рекомендации")
            ws_recs.append(["Специальность", "Рекомендация"])
            for spec in spec_names:
                try:
                    from hh_competency.methodology.fgos import FgosComparator

                    fgos_comparator = FgosComparator(database)
                    recs = fgos_comparator.generate_recommendations(spec)
                    if isinstance(recs, list):
                        for rec in recs:
                            ws_recs.append([spec, rec])
                except ImportError:
                    ws_recs.append([spec, "Модуль ФГОС не загружен"])
                except Exception:
                    ws_recs.append([spec, "Ошибка при получении рекомендаций"])

            wb.save(xlsx_path)
            console.print(f"[green]Excel saved to:[/] {os.path.abspath(xlsx_path)}")

        else:
            console.print(f"[red]Unknown format: {format}. Use 'excel' or 'csv'.[/]")
            raise typer.Exit(code=1)

    finally:
        database.close()


# ---------------------------------------------------------------------------
# report summary
# ---------------------------------------------------------------------------


@report_app.command(name="summary")
def report_summary(
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Show a cross-specialty summary of all analysis dimensions.

    Displays vacancy counts, top skills, discovered roles, overlap matrix,
    and FGOS gap analysis in a single Rich-formatted terminal report.

    Example: hh-competency report summary
    """
    app_config = load_config(config)
    database = _build_db(app_config, db)

    try:
        specialties = sorted(app_config.specialties.keys())
        console.print("\n[bold blue]═══ hh-competency Summary ═══[/]\n")
        console.print(f"Specialties: [cyan]{len(specialties)}[/]")
        console.print(f"Database: [dim]{database.db_path}[/]\n")

        # --- 1. Vacancy counts ---
        console.print("[bold]1. Vacancy Counts[/]")
        vc_table = Table()
        vc_table.add_column("Specialty", style="cyan")
        vc_table.add_column("Vacancies", justify="right", style="green")
        vc_table.add_column("Skills", justify="right", style="yellow")

        total_vacancies = 0
        for spec in specialties:
            vc = database.get_vacancy_count(spec)
            sc = len(database.get_skill_frequencies(spec, top_n=500))
            vc_table.add_row(spec, str(vc), str(sc))
            total_vacancies += vc

        vc_table.add_section()
        vc_table.add_row("[bold]Total[/]", f"[bold]{total_vacancies}[/]", "")
        console.print(vc_table)

        # --- 2. Top skills ---
        console.print("\n[bold]2. Top Skills per Specialty[/]")
        skills_table = Table()
        skills_table.add_column("Specialty", style="cyan")
        skills_table.add_column("Top 5 Skills", style="green")

        from hh_competency.analysis.skills import SkillAnalyzer
        from hh_competency.nlp.pipeline import NLPPipeline

        pipeline = NLPPipeline()
        analyzer = SkillAnalyzer(database, pipeline)

        for spec in specialties:
            top5 = analyzer.top_skills(spec, n=5)
            skills_str = (
                ", ".join(s.lemma for s in top5)
                if top5
                else "[dim](no data)[/]"
            )
            skills_table.add_row(spec, skills_str)

        console.print(skills_table)

        # --- 3. Role discovery ---
        console.print("\n[bold]3. Discovered Roles[/]")
        from hh_competency.analysis.clustering import SkillClusterer

        clusterer = SkillClusterer(database)
        total_roles = 0
        for spec in specialties:
            try:
                roles = clusterer.discover_roles(spec)
                if roles:
                    role_names = ", ".join(r.role_name for r in roles)
                    console.print(f"  [cyan]{spec}[/]: {role_names}")
                    total_roles += len(roles)
                else:
                    console.print(f"  [dim]{spec}: no roles discovered[/]")
            except Exception:
                console.print(f"  [dim]{spec}: (error discovering roles)[/]")

        # --- 4. Overlap matrix ---
        if len(specialties) >= 2:
            console.print("\n[bold]4. Skill Overlap (Jaccard)[/]")
            from hh_competency.analysis.compare import SkillComparator

            comparator = SkillComparator(database)
            matrix = comparator.overlap_matrix(specialties)

            om_table = Table(title="Jaccard Similarity Matrix")
            om_table.add_column("", style="dim")
            for spec in specialties:
                om_table.add_column(spec, justify="right")

            for spec_row in specialties:
                row = [spec_row]
                for spec_col in specialties:
                    val = matrix.get(spec_row, {}).get(spec_col, 0.0)
                    if spec_row == spec_col:
                        row.append("[dim]1.00[/]")
                    elif val > 0.3:
                        row.append(f"[green]{val:.2f}[/]")
                    elif val > 0.1:
                        row.append(f"[yellow]{val:.2f}[/]")
                    else:
                        row.append(f"[red]{val:.2f}[/]")
                om_table.add_row(*row)

            console.print(om_table)

        # --- 5. FGOS gaps ---
        console.print("\n[bold]5. FGOS Gaps[/]")
        try:
            from hh_competency.methodology.fgos import FgosComparator

            fgos_comparator = FgosComparator(database)
            for spec in specialties:
                try:
                    rec = fgos_comparator.compare_specialty(spec)
                    status = rec.get("status", "unknown")
                    console.print(f"  [cyan]{spec}[/]: {status}")
                except Exception:
                    console.print(f"  [dim]{spec}: (error in FGOS comparison)[/]")
        except ImportError:
            console.print(
                "  [dim]FGOS comparison module not available[/]"
            )

        # --- 6. Profstandart mapping ---
        console.print("\n[bold]6. Profstandart Mapping[/]")
        try:
            from hh_competency.methodology.profstandart import ProfstandartMapper

            ps_mapper = ProfstandartMapper(database)
            for spec in specialties:
                try:
                    ps_result = ps_mapper.map_roles_to_profstandart(spec)
                    if ps_result:
                        ps_count = len(ps_result)
                        console.print(
                            f"  [cyan]{spec}[/]: {ps_count} mapping(s)"
                        )
                    else:
                        console.print(f"  [dim]{spec}: no mappings[/]")
                except Exception:
                    console.print(f"  [dim]{spec}: (error in mapping)[/]")
        except ImportError:
            console.print(
                "  [dim]Profstandart mapping module not available[/]"
            )

        # --- 7. Curriculum recommendations ---
        console.print("\n[bold]7. Curriculum Recommendations[/]")
        try:
            from hh_competency.methodology.fgos import FgosComparator

            fgos_comparator = FgosComparator(database)
            for spec in specialties:
                try:
                    recs = fgos_comparator.generate_recommendations(spec)
                    if isinstance(recs, list) and recs:
                        console.print(f"\n  [bold cyan]{spec}[/]:")
                        for rec in recs[:5]:
                            console.print(f"    • {rec}")
                    else:
                        console.print(f"  [dim]{spec}: no recommendations[/]")
                except Exception:
                    console.print(f"  [dim]{spec}: (error)[/]")
        except ImportError:
            console.print(
                "  [dim]FGOS module not available. "
                "Install and configure methodology/fgos.py for recommendations.[/]"
            )

        console.print("\n[bold green]═══ Summary complete ═══[/]\n")

    finally:
        database.close()
