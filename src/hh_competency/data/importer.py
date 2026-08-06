"""Data import framework: load external datasets into DuckDB pipeline.

Supports CSV, JSONL, and HF datasets. All sources normalize to VacancyData
and feed into the same analysis pipeline as HH.ru scraped data.

IT vacancy filtering: vacancies matching known IT titles or containing
IT-specific technology keywords (Java, React, Docker, etc.) are excluded.
Python, C++, MATLAB, R are NOT considered IT — they pass through.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from hh_competency.nlp.stopwords import StopwordFilter
from hh_competency.storage.db import Database
from hh_competency.storage.models import ScrapeRun, VacancyData

# ───────────────────────────────────────────────────────────────────
# IT vacancy exclusion patterns
# ───────────────────────────────────────────────────────────────────

# Title regex: vacancy titles that are inherently IT/software engineering.
# These are applied IN ADDITION TO keyword-based is_it_vacancy() check.
_EXCLUDE_TITLES = re.compile(
    r"\b(?:"
    r"разработчик|программист|developer|programmer"
    r"|software engineer|software developer"
    r"|web developer|mobile developer"
    r"|devops engineer|devops"
    r"|qa engineer|qa инженер|тестировщик"
    r"|системный администратор|system administrator"
    r"|сетевой инженер|network engineer"
    r"|технический директор|cto|chief technology officer"
    r")\b",
    re.IGNORECASE,
)

# Lazily-initialized IT keyword filter singleton.
_IT_FILTER: StopwordFilter | None = None


def _get_it_filter() -> StopwordFilter:
    """Return the singleton StopwordFilter for IT exclusion checks."""
    global _IT_FILTER
    if _IT_FILTER is None:
        _IT_FILTER = StopwordFilter()
    return _IT_FILTER


def is_it_vacancy_normalized(name: str, description: str | None = None) -> bool:
    """Check if a vacancy should be excluded as pure IT.

    Two-stage filter (OR logic):
    1. Title matches EXCLUDE_TITLES regex (e.g. "Python разработчик")
    2. Keyword-based is_it_vacancy() check (e.g. "Java" in description)

    Python, C++, MATLAB, R in the description do NOT trigger exclusion.
    """
    if _EXCLUDE_TITLES.search(name):
        return True
    return _get_it_filter().is_it_vacancy(name, description)


class DataSource(Protocol):
    """Protocol for pluggable data sources.

    Any object with a `load() -> list[VacancyData]` method can be used.
    """

    def load(self) -> list[VacancyData]: ...

    @property
    def source_label(self) -> str: ...


class ImportResult:
    """Result of a data import operation."""

    def __init__(
        self,
        source_label: str,
        total_rows: int = 0,
        parsed: int = 0,
        skipped: int = 0,
        new_vacancies: int = 0,
        skipped_existing: int = 0,
        errors: list[str] | None = None,
        specialty: str = "imported",
    ):
        self.source_label = source_label
        self.total_rows = total_rows
        self.parsed = parsed
        self.skipped = skipped
        self.new_vacancies = new_vacancies
        self.skipped_existing = skipped_existing
        self.errors = errors or []
        self.specialty = specialty

    @property
    def success(self) -> bool:
        return self.new_vacancies > 0

    def summary(self) -> str:
        lines = [
            f"Source: {self.source_label}",
            f"Specialty: {self.specialty}",
            f"Total rows: {self.total_rows}",
            f"Parsed: {self.parsed}",
            f"Skipped (invalid): {self.skipped}",
            f"New: {self.new_vacancies}",
            f"Already in DB: {self.skipped_existing}",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


class DataImporter:
    """Orchestrates data import from external sources into DuckDB."""

    def __init__(self, db: Database, specialty: str = "imported"):
        self._db = db
        self._specialty = specialty

    def import_source(self, source: DataSource) -> ImportResult:
        """Load data from a source and insert into DuckDB.

        Deduplicates by vacancy ID: if a vacancy with the same ID already
        exists, it is skipped.
        """
        result = ImportResult(
            source_label=source.source_label,
            specialty=self._specialty,
        )

        try:
            vacancies = source.load()
            result.total_rows = len(vacancies)
            result.parsed = result.total_rows
        except Exception as exc:
            result.errors.append(f"Load failed: {exc}")
            return result

        # Deduplicate: skip vacancies already in the database
        existing_ids = self._db.get_existing_ids([v.id for v in vacancies])
        new_vacancies = [v for v in vacancies if v.id not in existing_ids]
        result.skipped_existing = result.parsed - len(new_vacancies)

        if not new_vacancies:
            return result

        # Create a ScrapeRun to group this import
        run_id = str(uuid.uuid4())
        run = ScrapeRun(
            run_id=run_id,
            specialty=self._specialty,
            query_params={"source": source.source_label},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            vacancies_found=len(new_vacancies),
            vacancies_fetched=len(new_vacancies),
        )
        self._db.insert_run(run)

        try:
            self._db.insert_vacancies(run_id, new_vacancies)
            result.new_vacancies = len(new_vacancies)
        except Exception as exc:
            result.errors.append(f"Insert failed: {exc}")
            result.new_vacancies = 0

        return result

    def import_file(
        self,
        path: str | Path,
        source_label: str | None = None,
        format: str = "auto",
        column_map: dict[str, str] | None = None,
    ) -> ImportResult:
        """Import from a file, auto-detecting format.

        Args:
            path: Path to the data file.
            source_label: Human-readable label for logging. Auto-generated if None.
            format: 'auto', 'csv', 'jsonl'. Auto-detects by extension.
            column_map: Map column names from file to VacancyData fields.
                        Defaults to standard HH.ru API field names.

        Returns:
            ImportResult with insertion statistics.
        """
        path = Path(path)
        if source_label is None:
            source_label = f"{path.name}"

        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".csv":
                format = "csv"
            elif suffix in (".jsonl", ".json"):
                format = "jsonl"
            else:
                return ImportResult(
                    source_label=source_label,
                    errors=[f"Unknown format for {suffix}. Use --format csv|jsonl"],
                )

        if format == "csv":
            from hh_competency.data.sources.csv_source import CSVSource

            source = CSVSource(path, source_label=source_label, column_map=column_map)
        elif format == "jsonl":
            from hh_competency.data.sources.jsonl_source import JSONLSource

            source = JSONLSource(path, source_label=source_label, column_map=column_map)
        else:
            return ImportResult(
                source_label=source_label,
                errors=[f"Unsupported format: {format}"],
            )

        return self.import_source(source)


# ───────────────────────────────────────────────────────────────────
# Helpers: field normalization for raw data from various sources
# ───────────────────────────────────────────────────────────────────


def _strip_html(text: str | None) -> str | None:
    """Remove HTML tags from text fields (common in scraped data)."""
    if not text:
        return text
    import re

    return re.sub(r"<[^>]+>", "", text)


def _normalize_salary(raw: dict | None) -> dict | None:
    """Normalize salary fields: ensure from/to are numbers, strip currency."""
    if not raw or not isinstance(raw, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("from", "to", "currency"):
        if key in raw and raw[key] is not None:
            result[key] = raw[key]
    return result if result else None


def _normalize_key_skills(raw: list | str | None) -> list[dict]:
    """Normalize key_skills: accept list of dicts, list of strings, or comma-separated."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [{"name": s.strip()} for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                result.append({"name": item.strip()})
        return result
    return []


def normalize_vacancy(
    raw: dict,
    column_map: dict[str, str] | None = None,
    filter_it: bool = True,
) -> VacancyData | None:
    """Convert a raw dict from any source into a standardized VacancyData.

    Args:
        raw: Raw dict from CSV row, JSONL line, or API response.
        column_map: Maps file column names → VacancyData field names.
            Default mapping matches HH.ru API field names directly.
        filter_it: If True, skip vacancies detected as pure IT (exclude
            titles matching EXCLUDE_TITLES regex and keyword-based checks).
            Set to False to import all vacancies unfiltered.

    Returns:
        VacancyData or None if required fields are missing or filtered out.
    """
    if column_map is None:
        column_map = {
            "id": "id",
            "name": "name",
            "description": "description",
            "key_skills": "key_skills",
            "salary": "salary",
            "experience": "experience",
            "area": "area",
            "professional_roles": "professional_roles",
            "employer": "employer",
            "published_at": "published_at",
        }

    def _get(key: str) -> Any:
        file_col = column_map.get(key, key)  # type: ignore[arg-type]
        return raw.get(file_col)

    vacancy_id = _get("id")
    if not vacancy_id:
        return None
    vacancy_name = _get("name")
    if not vacancy_name:
        return None

    if filter_it and is_it_vacancy_normalized(
        str(vacancy_name), _strip_html(_get("description"))
    ):
        return None

    return VacancyData(
        id=str(vacancy_id),
        name=str(vacancy_name),
        description=_strip_html(_get("description")),
        key_skills=_normalize_key_skills(_get("key_skills")),
        salary=_normalize_salary(_get("salary")),
        experience=_get("experience"),
        area=_get("area"),
        professional_roles=_get("professional_roles") or [],
        employer=_get("employer"),
        published_at=_get("published_at"),
    )
