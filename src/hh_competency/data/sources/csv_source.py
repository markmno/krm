"""CSV data source: import vacancies from CSV files.

Supports both standard HH.ru API exports and arbitrary CSV with column mapping.
"""

from __future__ import annotations

import csv
import json
from contextlib import suppress
from pathlib import Path

from hh_competency.data.importer import normalize_vacancy
from hh_competency.storage.models import VacancyData


class CSVSource:
    """Load vacancies from a CSV file."""

    def __init__(
        self,
        path: str | Path,
        source_label: str | None = None,
        column_map: dict[str, str] | None = None,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ):
        self._path = Path(path)
        self.source_label = source_label or f"CSV: {self._path.name}"
        self._column_map = column_map
        self._encoding = encoding
        self._delimiter = delimiter

    def load(self) -> list[VacancyData]:
        """Read CSV rows and normalize to VacancyData objects."""
        vacancies: list[VacancyData] = []

        with self._path.open(encoding=self._encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self._delimiter)
            for row in reader:
                # Parse JSON-like fields that might be embedded in CSV
                row_parsed = dict(row)
                for key in list(row_parsed.keys()):
                    val = row_parsed[key]
                    if isinstance(val, str) and (val.startswith("{") or val.startswith("[")):
                        with suppress(json.JSONDecodeError, ValueError):
                            row_parsed[key] = json.loads(val)

                vac = normalize_vacancy(row_parsed, column_map=self._column_map)
                if vac is not None:
                    vacancies.append(vac)

        return vacancies
