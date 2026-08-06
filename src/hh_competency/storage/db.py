"""DuckDB storage layer for hh-competency.

Schema: scrape_runs, raw_vacancies, skill_frequencies, specialty_profiles.

Vacancy data stored as JSON for schema flexibility.
Analysis results stored in normalized tables for fast queries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from hh_competency.storage.models import (
    RoleProfile,
    ScrapeRun,
    SkillFrequency,
    VacancyData,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scrape_runs (
    run_id TEXT PRIMARY KEY,
    specialty TEXT NOT NULL,
    query_params JSON,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    vacancies_found INTEGER DEFAULT 0,
    vacancies_fetched INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS raw_vacancies (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    data TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skill_frequencies (
    specialty TEXT NOT NULL,
    lemma TEXT NOT NULL,
    pos TEXT NOT NULL,
    frequency INTEGER NOT NULL DEFAULT 0,
    vacancy_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (specialty, lemma)
);

CREATE TABLE IF NOT EXISTS specialty_profiles (
    specialty TEXT NOT NULL,
    role_name TEXT NOT NULL,
    defining_skills JSON,
    supporting_skills JSON,
    generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (specialty, role_name)
);
"""


class Database:
    """DuckDB wrapper for hh-competency data storage and queries.

    Supports both read-write (scrape phase) and read-only (analysis phase) modes.
    """

    def __init__(self, db_path: str | Path, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.db_path), read_only=read_only)
        if not read_only:
            self._con.execute(SCHEMA_SQL)

    # -- Scrape write operations --

    def insert_run(self, run: ScrapeRun) -> None:
        """Insert a new scrape run record."""
        self._con.execute(
            """
            INSERT INTO scrape_runs (run_id, specialty, query_params, started_at)
            VALUES (?, ?, ?, ?)
            """,
            [run.run_id, run.specialty, json.dumps(run.query_params), run.started_at],
        )

    def update_run_completed(
        self, run_id: str, vacancies_found: int, vacancies_fetched: int
    ) -> None:
        """Mark a scrape run as completed with final counts."""
        self._con.execute(
            """
            UPDATE scrape_runs
            SET completed_at = CURRENT_TIMESTAMP,
                vacancies_found = ?,
                vacancies_fetched = ?
            WHERE run_id = ?
            """,
            [vacancies_found, vacancies_fetched, run_id],
        )

    def insert_vacancies(self, run_id: str, vacancies: list[VacancyData]) -> None:
        """Batch insert vacancies for a scrape run."""
        rows = [
            (v.id, run_id, msgspec_json(v), v.published_at or datetime.now(UTC).isoformat())
            for v in vacancies
        ]
        self._con.executemany(
            """
            INSERT OR IGNORE INTO raw_vacancies (id, run_id, data, fetched_at)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )

    def vacancy_exists(self, vacancy_id: str) -> bool:
        """Check if a vacancy has already been scraped."""
        result = self._con.execute(
            "SELECT 1 FROM raw_vacancies WHERE id = ?", [vacancy_id]
        )
        return result.fetchone() is not None

    def get_existing_ids(self, vacancy_ids: list[str]) -> set[str]:
        """Return which of the given vacancy IDs already exist in the database."""
        if not vacancy_ids:
            return set()
        placeholders = ",".join(["?"] * len(vacancy_ids))
        result = self._con.execute(
            f"SELECT id FROM raw_vacancies WHERE id IN ({placeholders})",
            vacancy_ids,
        )
        return {row[0] for row in result.fetchall()}

    # -- Analysis read operations --

    def get_vacancy_descriptions(self, specialty: str) -> list[str]:
        """Extract description text from all vacancies for a specialty.

        Returns non-null descriptions only, joined from raw_vacancies via
        scrape_runs filtered by specialty.
        """
        result = self._con.execute(
            """
            SELECT json_extract_string(rv.data, '$.description')
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
              AND json_extract_string(rv.data, '$.description') IS NOT NULL
            """,
            [specialty],
        )
        return [row[0] for row in result.fetchall() if row[0]]

    def insert_skill_frequencies(self, frequencies: list[SkillFrequency]) -> None:
        """Batch upsert skill frequencies."""
        rows = [(f.specialty, f.lemma, f.pos, f.frequency, f.vacancy_count) for f in frequencies]
        self._con.executemany(
            """
            INSERT OR REPLACE INTO skill_frequencies
                (specialty, lemma, pos, frequency, vacancy_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    def get_skill_frequencies(
        self, specialty: str, top_n: int = 50
    ) -> list[SkillFrequency]:
        """Get top N skills for a specialty by frequency."""
        result = self._con.execute(
            """
            SELECT specialty, lemma, pos, frequency, vacancy_count
            FROM skill_frequencies
            WHERE specialty = ?
            ORDER BY frequency DESC
            LIMIT ?
            """,
            [specialty, top_n],
        )
        return [SkillFrequency(*row) for row in result.fetchall()]

    def get_skill_frequencies_multi(
        self, specialties: list[str]
    ) -> list[SkillFrequency]:
        """Get skill frequencies for multiple specialties."""
        if not specialties:
            return []
        placeholders = ",".join(["?"] * len(specialties))
        result = self._con.execute(
            f"""
            SELECT specialty, lemma, pos, frequency, vacancy_count
            FROM skill_frequencies
            WHERE specialty IN ({placeholders})
            ORDER BY specialty, frequency DESC
            """,
            specialties,
        )
        return [SkillFrequency(*row) for row in result.fetchall()]

    def insert_role_profiles(self, profiles: list[RoleProfile]) -> None:
        """Batch upsert role profiles."""
        rows = [
            (
                p.specialty,
                p.role_name,
                json.dumps(p.defining_skills),
                json.dumps(p.supporting_skills),
                p.generated_at,
            )
            for p in profiles
        ]
        self._con.executemany(
            """
            INSERT OR REPLACE INTO specialty_profiles
                (specialty, role_name, defining_skills, supporting_skills, generated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    def get_role_profiles(self, specialty: str) -> list[RoleProfile]:
        """Get all role profiles for a specialty."""
        import json

        result = self._con.execute(
            """
            SELECT specialty, role_name, defining_skills,
                   supporting_skills, generated_at
            FROM specialty_profiles
            WHERE specialty = ?
            ORDER BY role_name
            """,
            [specialty],
        )
        profiles = []
        for row in result.fetchall():
            defining = json.loads(row[2]) if isinstance(row[2], str) else row[2]
            supporting = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            profiles.append(
                RoleProfile(
                    specialty=row[0],
                    role_name=row[1],
                    defining_skills=[
                        (str(item[0]), float(item[1])) for item in defining
                    ],
                    supporting_skills=[
                        (str(item[0]), float(item[1])) for item in supporting
                    ],
                    generated_at=row[4],
                )
            )
        return profiles

    def get_scrape_runs(self, specialty: str | None = None) -> list[ScrapeRun]:
        """Get scrape runs, optionally filtered by specialty."""
        if specialty:
            result = self._con.execute(
                """
                SELECT run_id, specialty, query_params, started_at,
                       completed_at, vacancies_found, vacancies_fetched
                FROM scrape_runs
                WHERE specialty = ?
                ORDER BY started_at DESC
                """,
                [specialty],
            )
        else:
            result = self._con.execute(
                """
                SELECT run_id, specialty, query_params, started_at,
                       completed_at, vacancies_found, vacancies_fetched
                FROM scrape_runs
                ORDER BY started_at DESC
                """
            )
        rows = result.fetchall()
        return [
            ScrapeRun(
                run_id=row[0],
                specialty=row[1],
                query_params=json.loads(row[2]) if row[2] else {},
                started_at=row[3],
                completed_at=row[4],
                vacancies_found=row[5],
                vacancies_fetched=row[6],
            )
            for row in rows
        ]

    def get_vacancy_count(self, specialty: str) -> int:
        """Count vacancies for a specialty."""
        result = self._con.execute(
            """
            SELECT COUNT(*)
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
            """,
            [specialty],
        )
        return result.fetchone()[0]

    def get_salary_data(self, specialty: str) -> pd.DataFrame:
        """Extract salary data for a specialty as a pandas DataFrame."""
        return self._con.execute(
            """
            SELECT
                json_extract_string(rv.data, '$.salary.from')::FLOAT AS salary_from,
                json_extract_string(rv.data, '$.salary.to')::FLOAT AS salary_to,
                json_extract_string(rv.data, '$.salary.currency') AS currency,
                json_extract_string(rv.data, '$.experience.name') AS experience,
                json_extract_string(rv.data, '$.area.name') AS area
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
              AND json_extract_string(rv.data, '$.salary.from') IS NOT NULL
            """,
            [specialty],
        ).df()

    # ── Yearly dynamics ──────────────────────────────────────────────

    def get_available_years(self, specialty: str) -> list[int]:
        """Return sorted list of years for which vacancy data exists."""
        result = self._con.execute(
            """
            SELECT DISTINCT
                CAST(SUBSTR(json_extract_string(rv.data, '$.published_at'), 1, 4) AS INTEGER) AS yr
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
              AND json_extract_string(rv.data, '$.published_at') IS NOT NULL
            ORDER BY yr
            """,
            [specialty],
        )
        return [row[0] for row in result.fetchall() if row[0]]

    def get_vacancies_by_year(self, specialty: str) -> dict[int, list[str]]:
        """Get vacancy descriptions grouped by year.

        Returns:
            {2023: [desc1, desc2, ...], 2024: [desc3, ...], ...}
        """
        result = self._con.execute(
            """
            SELECT
                CAST(SUBSTR(json_extract_string(rv.data, '$.published_at'), 1, 4) AS INTEGER) AS yr,
                json_extract_string(rv.data, '$.description') AS desc
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
              AND json_extract_string(rv.data, '$.published_at') IS NOT NULL
              AND json_extract_string(rv.data, '$.description') IS NOT NULL
            ORDER BY yr
            """,
            [specialty],
        )
        by_year: dict[int, list[str]] = {}
        for yr, desc in result.fetchall():
            if yr and desc:
                by_year.setdefault(yr, []).append(desc)
        return by_year

    def get_yearly_salary_data(self, specialty: str) -> dict[int, list[dict]]:
        """Get salary records grouped by year."""
        result = self._con.execute(
            """
            SELECT
                CAST(SUBSTR(json_extract_string(rv.data, '$.published_at'), 1, 4) AS INTEGER) AS yr,
                json_extract_string(rv.data, '$.salary.from')::FLOAT AS salary_from,
                json_extract_string(rv.data, '$.salary.to')::FLOAT AS salary_to,
                json_extract_string(rv.data, '$.salary.currency') AS currency
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
              AND json_extract_string(rv.data, '$.published_at') IS NOT NULL
              AND json_extract_string(rv.data, '$.salary.from') IS NOT NULL
            ORDER BY yr
            """,
            [specialty],
        )
        by_year: dict[int, list[dict]] = {}
        for yr, s_from, s_to, currency in result.fetchall():
            if yr and s_from is not None:
                by_year.setdefault(yr, []).append(
                    {"from": s_from, "to": s_to, "currency": currency}
                )
        return by_year

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def get_vacancy_names(self, specialty: str, limit: int = 100) -> list[str]:
        """Return list of unique vacancy names for a specialty."""
        result = self._con.execute(
            """
            SELECT DISTINCT json_extract_string(rv.data, '$.name') AS name
            FROM raw_vacancies rv
            JOIN scrape_runs sr ON rv.run_id = sr.run_id
            WHERE sr.specialty = ?
              AND json_extract_string(rv.data, '$.name') IS NOT NULL
            LIMIT ?
            """,
            [specialty, limit],
        )
        return [row[0] for row in result.fetchall() if row[0]]

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Raw DuckDB connection for advanced queries."""
        return self._con


def msgspec_json(obj: VacancyData) -> str:
    """Serialize VacancyData to JSON string using msgspec.

    Uses a local import to avoid circular dependencies.
    """
    import msgspec

    return msgspec.json.encode(obj).decode("utf-8")
