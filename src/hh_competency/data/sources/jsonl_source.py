"""JSONL data source: import vacancies from newline-delimited JSON files.

Most scraped data dumps (including Wayback Machine captures) use JSONL format.
"""

from __future__ import annotations

import json
from pathlib import Path

from hh_competency.data.importer import normalize_vacancy
from hh_competency.storage.models import VacancyData


class JSONLSource:
    """Load vacancies from a JSONL (newline-delimited JSON) file."""

    def __init__(
        self,
        path: str | Path,
        source_label: str | None = None,
        column_map: dict[str, str] | None = None,
    ):
        self._path = Path(path)
        self.source_label = source_label or f"JSONL: {self._path.name}"
        self._column_map = column_map

    def load(self) -> list[VacancyData]:
        """Read JSONL lines and normalize to VacancyData objects."""
        vacancies: list[VacancyData] = []

        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                vac = normalize_vacancy(raw, column_map=self._column_map)
                if vac is not None:
                    vacancies.append(vac)

        return vacancies
