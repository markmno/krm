"""hh-competency: Build competency-role models from HH.ru vacancy data."""

from __future__ import annotations

import typer
from rich.console import Console

from hh_competency.commands.analyze import analyze_app
from hh_competency.commands.data import data_app
from hh_competency.commands.report import report_app
from hh_competency.commands.scrape import scrape_app
from hh_competency.commands.workflow import workflow_app

__version__ = "0.1.0"

app = typer.Typer(
    help="hh-competency: Competency-role model discovery from Russian job market data"
)


@app.callback(invoke_without_command=True)
def _version_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit"
    ),
) -> None:
    """Print version information and show help if no command given."""
    if version:
        console = Console()
        console.print(f"[bold cyan]hh-competency[/] v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console = Console()
        console.print("[bold cyan]hh-competency[/] v0.1.0")
        console.print(
            "Competency-role model discovery from Russian job market data\n"
        )
        console.print("Usage: hh-competency [OPTIONS] COMMAND [ARGS]...\n")
        console.print("[bold]Commands:[/]")
        console.print("  [cyan]scrape[/]   Scrape vacancy data from HH.ru API")
        console.print("  [cyan]data[/]     Import from external datasets (CSV, JSONL)")
        console.print("  [cyan]analyze[/]  Analyze vacancy data: skills, roles, stats, simulation")
        console.print("  [cyan]report[/]   Generate reports, export data, view summaries")
        console.print(
            "  [cyan]workflow[/] Full ITMO pipeline: scrape→analyze→stats→simulate→report"
        )
        raise typer.Exit()


app.add_typer(scrape_app, name="scrape")
app.add_typer(data_app, name="data")
app.add_typer(analyze_app, name="analyze")
app.add_typer(report_app, name="report")
app.add_typer(workflow_app, name="workflow")
