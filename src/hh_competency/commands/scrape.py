"""Scrape engine and CLI commands for HH.ru vacancy data collection.

Orchestrates search → paginate → deduplicate → fetch details → store workflow.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from hh_competency.api.client import HHClient
from hh_competency.config import load_config
from hh_competency.storage.db import Database
from hh_competency.storage.models import AppConfig, ScrapeRun, VacancyData

console = Console()
scrape_app = typer.Typer(help="Scrape vacancy data from HH.ru API")


class ScrapeEngine:
    """Orchestrates the full scrape workflow with progress tracking and resume support."""

    def __init__(self, db: Database, config: AppConfig) -> None:
        self._db = db
        self._config = config

    async def scrape_specialty(
        self,
        client: HHClient,
        specialty_name: str,
        max_vacancies: int = 500,
        workers: int = 3,
    ) -> ScrapeRun:
        """Scrape all vacancies for a single specialty.

        Workflow: lookup config → search all keywords → deduplicate IDs →
        skip existing → fetch details → store.
        """
        spec = self._config.specialties.get(specialty_name)
        if spec is None:
            available = ", ".join(sorted(self._config.specialties.keys()))
            msg = f"Unknown specialty '{specialty_name}'. Available: {available}"
            raise typer.BadParameter(msg)

        run_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)
        area = 113  # Россия

        run = ScrapeRun(
            run_id=run_id,
            specialty=specialty_name,
            query_params={
                "search_keywords": spec.search_keywords,
                "professional_roles": spec.professional_roles,
                "area": area,
            },
            started_at=started_at,
        )

        console.print(
            f"[bold blue]Scraping specialty:[/] {specialty_name} "
            f"({len(spec.search_keywords)} keywords)"
        )
        self._db.insert_run(run)

        # Phase 1: Search all keywords, collect unique vacancy IDs and briefs
        seen_ids: set[str] = set()
        vacancy_briefs: list[dict[str, Any]] = []

        for keyword in spec.search_keywords:
            console.print(f"  [dim]Searching: '{keyword}'[/]")
            for page in range(20):  # max_pages per keyword
                search_result = await client.search_vacancies(
                    text=keyword,
                    area=area,
                    per_page=100,
                    page=page,
                )
                items = search_result.get("items", [])
                if not items:
                    break

                for item in items:
                    vac_id = str(item["id"])
                    if vac_id not in seen_ids:
                        seen_ids.add(vac_id)
                        vacancy_briefs.append(item)

                total_pages = search_result.get("pages", 0)
                if page + 1 >= total_pages:
                    break

        vacancies_found = len(vacancy_briefs)
        console.print(f"  [green]Found {vacancies_found} unique vacancies[/]")

        # Phase 2: Skip existing IDs (resume support)
        all_ids = [b["id"] for b in vacancy_briefs]
        existing_ids = self._db.get_existing_ids(all_ids) if all_ids else set()
        new_briefs = [b for b in vacancy_briefs if b["id"] not in existing_ids]

        if existing_ids:
            console.print(
                f"  [dim]Skipping {len(existing_ids)} already-scraped vacancies[/]"
            )

        if not new_briefs:
            console.print("  [yellow]No new vacancies to fetch[/]")
            self._db.update_run_completed(run_id, vacancies_found, 0)
            run.vacancies_found = vacancies_found
            run.vacancies_fetched = 0
            run.completed_at = datetime.now(UTC)
            return run

        # Phase 3: Fetch full details with semaphore-limited concurrency
        semaphore = asyncio.Semaphore(workers)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[cyan]Fetching details...", total=len(new_briefs)
            )

            async def _fetch_detail(brief: dict[str, Any]) -> VacancyData:
                async with semaphore:
                    detail = await client.get_vacancy_detail(brief["id"])
                    progress.advance(task_id)
                    return VacancyData.from_api_response(detail)

            # Apply max_vacancies limit
            briefs_to_fetch = new_briefs[:max_vacancies]
            vacancies = await asyncio.gather(
                *[_fetch_detail(b) for b in briefs_to_fetch]
            )

        vacancies_fetched = len(vacancies)

        # Phase 4: Store
        if vacancies:
            self._db.insert_vacancies(run_id, list(vacancies))
            console.print(f"  [green]Stored {vacancies_fetched} vacancies[/]")

        self._db.update_run_completed(run_id, vacancies_found, vacancies_fetched)
        run.vacancies_found = vacancies_found
        run.vacancies_fetched = vacancies_fetched
        run.completed_at = datetime.now(UTC)

        return run

    async def scrape_all(
        self,
        client: HHClient,
        max_vacancies: int = 500,
        workers: int = 3,
    ) -> list[ScrapeRun]:
        """Scrape all configured specialties sequentially."""
        specialty_names = sorted(self._config.specialties.keys())
        console.print(
            f"[bold blue]Scraping {len(specialty_names)} specialties[/]\n"
        )

        runs: list[ScrapeRun] = []
        for i, name in enumerate(specialty_names, 1):
            console.print(f"\n[bold]--- [{i}/{len(specialty_names)}] ---[/]")
            run = await self.scrape_specialty(
                client, name, max_vacancies=max_vacancies, workers=workers
            )
            runs.append(run)

        # Summary table
        self._print_summary(runs)
        return runs

    @staticmethod
    def _merge_and_dedupe(results: list[VacancyData]) -> list[VacancyData]:
        """Deduplicate vacancy data list by vacancy ID, keeping first occurrence."""
        seen: set[str] = set()
        deduped: list[VacancyData] = []
        for vac in results:
            if vac.id not in seen:
                seen.add(vac.id)
                deduped.append(vac)
        return deduped

    @staticmethod
    def _print_summary(runs: list[ScrapeRun]) -> None:
        """Print a Rich table summary of scrape runs."""
        table = Table(title="Scrape Summary")
        table.add_column("Specialty", style="cyan")
        table.add_column("Found", justify="right", style="green")
        table.add_column("Fetched", justify="right", style="green")
        table.add_column("Duration", justify="right", style="yellow")

        for run in runs:
            duration = ""
            if run.completed_at:
                delta = run.completed_at - run.started_at
                seconds = int(delta.total_seconds())
                duration = f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60}s"
            table.add_row(
                run.specialty,
                str(run.vacancies_found),
                str(run.vacancies_fetched),
                duration,
            )

        console.print(table)


# ---------------------------------------------------------------------------
# Helper: build engine + client from CLI options
# ---------------------------------------------------------------------------


def _build_engine_and_client(
    config_path: str,
    db_path: str,
) -> tuple[ScrapeEngine, AppConfig]:
    """Load config and create a ScrapeEngine (without HHClient, which needs async context)."""
    config = load_config(config_path)
    db = Database(db_path or config.database_path)
    return ScrapeEngine(db, config), config


def _make_client(config: AppConfig) -> HHClient:
    """Create an HHClient from AppConfig settings."""
    return HHClient(
        base_url=config.hh_api_base_url,
        user_agent=config.hh_api_user_agent,
        requests_per_second=config.hh_api_requests_per_second,
        max_retries=config.hh_api_max_retries,
        retry_backoff=config.hh_api_retry_backoff,
    )


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


@scrape_app.command(name="specialty")
def scrape_specialty_cmd(
    specialty: str = typer.Argument(..., help="Specialty name to scrape"),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
    max_vacancies: int = typer.Option(
        500, "--max-vacancies", "-n", help="Max vacancies to fetch"
    ),
    workers: int = typer.Option(
        3, "--workers", "-w", help="Concurrent detail fetches"
    ),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
) -> None:
    """Scrape a single specialty from the config.

    Example: hh-competency scrape specialty physics -n 200 -w 5
    """
    engine, app_config = _build_engine_and_client(config, db)

    async def _run() -> None:
        client = _make_client(app_config)
        async with client:
            run = await engine.scrape_specialty(
                client, specialty, max_vacancies=max_vacancies, workers=workers
            )
            engine._print_summary([run])

    asyncio.run(_run())


@scrape_app.command(name="all")
def scrape_all_cmd(
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
    max_vacancies: int = typer.Option(
        500, "--max-vacancies", "-n", help="Max vacancies per specialty"
    ),
    workers: int = typer.Option(
        3, "--workers", "-w", help="Concurrent detail fetches"
    ),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
) -> None:
    """Scrape all specialties defined in the config file.

    Example: hh-competency scrape all -n 300 -w 5
    """
    engine, app_config = _build_engine_and_client(config, db)

    async def _run() -> None:
        client = _make_client(app_config)
        async with client:
            await engine.scrape_all(
                client, max_vacancies=max_vacancies, workers=workers
            )

    asyncio.run(_run())


@scrape_app.command(name="search")
def scrape_search_cmd(
    query: str = typer.Argument(..., help="Search query text"),
    area: int = typer.Option(113, "--area", "-a", help="HH.ru area ID (113=Россия)"),
    pages: int = typer.Option(
        5, "--pages", "-p", help="Max pages to search (100 results/page)"
    ),
    db: str = typer.Option(
        "", "--db", "-d", help="Database path (default from config)"
    ),
    config: str = typer.Option(
        "specialties.yml", "--config", "-c", help="Path to specialties.yml"
    ),
) -> None:
    """Custom search without specialty config.

    Searches HH.ru, fetches full details, and stores results.

    Example: hh-competency scrape search "python разработчик" -a 1 -p 3
    """
    app_config = load_config(config)
    database = Database(db or app_config.database_path)

    async def _run() -> None:
        client = _make_client(app_config)
        async with client:
            run_id = str(uuid.uuid4())
            started_at = datetime.now(UTC)

            run = ScrapeRun(
                run_id=run_id,
                specialty=f"search:{query}",
                query_params={"text": query, "area": area, "pages": pages},
                started_at=started_at,
            )
            database.insert_run(run)

            # Phase 1: Search and collect briefs
            seen_ids: set[str] = set()
            vacancy_briefs: list[dict[str, Any]] = []

            console.print(f"[bold blue]Searching:[/] '{query}' in area {area}")
            for page in range(pages):
                search_result = await client.search_vacancies(
                    text=query, area=area, per_page=100, page=page
                )
                items = search_result.get("items", [])
                if not items:
                    break
                for item in items:
                    vac_id = str(item["id"])
                    if vac_id not in seen_ids:
                        seen_ids.add(vac_id)
                        vacancy_briefs.append(item)

                if page + 1 >= search_result.get("pages", 0):
                    break

            vacancies_found = len(vacancy_briefs)
            console.print(f"  [green]Found {vacancies_found} vacancies[/]")

            # Phase 2: Skip existing
            all_ids = [b["id"] for b in vacancy_briefs]
            existing_ids = database.get_existing_ids(all_ids) if all_ids else set()
            new_briefs = [b for b in vacancy_briefs if b["id"] not in existing_ids]

            if existing_ids:
                console.print(
                    f"  [dim]Skipping {len(existing_ids)} already-scraped[/]"
                )

            if not new_briefs:
                console.print("  [yellow]No new vacancies to fetch[/]")
                database.update_run_completed(run_id, vacancies_found, 0)
                return

            # Phase 3: Fetch details
            semaphore = asyncio.Semaphore(3)

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    "[cyan]Fetching details...", total=len(new_briefs)
                )

                async def _fetch_detail(brief: dict[str, Any]) -> VacancyData:
                    async with semaphore:
                        detail = await client.get_vacancy_detail(brief["id"])
                        progress.advance(task_id)
                        return VacancyData.from_api_response(detail)

                vacancies = await asyncio.gather(
                    *[_fetch_detail(b) for b in new_briefs]
                )

            # Phase 4: Store
            if vacancies:
                database.insert_vacancies(run_id, list(vacancies))
                console.print(f"  [green]Stored {len(vacancies)} vacancies[/]")

            vacancies_fetched = len(vacancies)
            database.update_run_completed(run_id, vacancies_found, vacancies_fetched)

            # Summary
            table = Table(title="Search Summary")
            table.add_column("Query", style="cyan")
            table.add_column("Found", justify="right", style="green")
            table.add_column("Fetched", justify="right", style="green")
            table.add_row(query, str(vacancies_found), str(vacancies_fetched))
            console.print(table)

    asyncio.run(_run())
