"""Data import CLI: load vacancies from external datasets and historical archives.

Supports CSV, JSONL, and HuggingFace datasets. All data feeds into the
same DuckDB pipeline as HH.ru scraped data.

Examples:
    hh-competency data import data/vacancies.jsonl --specialty physics
    hh-competency data import data/dump.csv --format csv --column-map "id=id,name=name,description=description"
    hh-competency data import --list-sources
    hh-competency data sources
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hh_competency.storage.db import Database

data_app = typer.Typer(help="Import data from external sources and historical archives")
console = Console()

KNOWN_SOURCES = [
    ("HuggingFace", "trewwxsav/IT_vacancies_from_hh.ru", "47MB, IT-only, HF dataset, pip install datasets"),
    ("HuggingFace", "russian-oracle/vacancy-section-classifier-ru", "18K labeled sentences, HF dataset"),
    ("HuggingFace", "liovina/vacancyradar-data", "Self-refreshing IT vacancy event store, DuckDB"),
    ("HuggingFace", "evilfreelancer/headhunter", "319 rows, small sample"),
    ("Trudvsem API", "https://trudvsem.ru/opendata", "Government job portal, NO AUTH, REST API + bulk CSV"),
    ("SuperJob API", "https://api.superjob.ru", "REST API v2.0, OAuth2, 120 req/min"),
    ("Wayback Machine", "https://web.archive.org/web/*/hh.ru/vacancies/*", "Historical HH.ru search page captures"),
    ("CSV dumps", "Any CSV with columns: id, name, description, ...", "Use --column-map for custom schemas"),
    ("JSONL dumps", "Newline-delimited JSON, one vacancy per line", "Most common for scraped data"),
]


@data_app.command("sources")
def list_sources() -> None:
    """List known external data sources for Russian job market data."""
    table = Table(title="External Data Sources")
    table.add_column("Type", style="cyan")
    table.add_column("Identifier", style="green")
    table.add_column("Notes")

    for src_type, identifier, notes in KNOWN_SOURCES:
        table.add_row(src_type, identifier, notes)

    console.print(table)
    console.print("\n[yellow]Note:[/] For HuggingFace datasets, install:")
    console.print("  uv add datasets")


@data_app.command("import")
def import_data(
    path: str = typer.Argument(..., help="Path to data file (CSV, JSONL)"),
    specialty: str = typer.Option("imported", "--specialty", "-s", help="Specialty label for imported data"),
    db_path: str = typer.Option("data/hh_competency.db", "--db", "-d", help="DuckDB database path"),
    format: str = typer.Option("auto", "--format", "-f", help="auto|csv|jsonl"),
    source_label: str = typer.Option("", "--label", "-l", help="Human-readable label for this import"),
    column_map: str = typer.Option(
        "",
        "--column-map",
        "-m",
        help="Column mapping: 'id=col_name,name=title_col,...'. Default: HH.ru API field names.",
    ),
) -> None:
    """Import vacancies from an external data file into the database.

    Supported formats: CSV, JSONL (newline-delimited JSON).

    The data is normalized to standard VacancyData format and deduplicated
    by vacancy ID — existing vacancies are not overwritten.

    Examples:
        hh-competency data import data/dump.jsonl --specialty physics
        hh-competency data import data/export.csv -m "id=vacancy_id,name=title,description=body"
    """
    from hh_competency.data.importer import DataImporter

    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found:[/] {path}")
        raise typer.Exit(code=1)

    # Parse column map
    parsed_map: dict[str, str] | None = None
    if column_map:
        parsed_map = {}
        for pair in column_map.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                parsed_map[k.strip()] = v.strip()

    db = Database(db_path, read_only=False)
    importer = DataImporter(db, specialty=specialty)

    label = source_label or file_path.name

    console.print(f"[bold]Importing:[/] {label}")
    console.print(f"  Format: {format}, Specialty: {specialty}")

    result = importer.import_file(
        path=file_path,
        source_label=label,
        format=format,
        column_map=parsed_map,
    )

    console.print("\n[bold]Result:[/]")
    console.print(result.summary())

    if result.success:
        console.print(f"\n[green]✓ Imported {result.new_vacancies} new vacancies[/]")
        console.print("[dim]Run 'hh-competency workflow run --skip-scrape' to analyze[/]")
    elif result.errors:
        for err in result.errors:
            console.print(f"[yellow]⚠ {err}[/]")
    else:
        console.print("[dim]All vacancies already in database[/]")

    db.close()
