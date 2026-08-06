"""Data models for hh-competency using msgspec."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from msgspec import Struct, field

# -- Configuration types --


class SpecialtyDefinition(Struct):
    """Configuration for a single specialty to scrape and analyze."""

    name: str
    search_keywords: list[str]
    nlp_keywords: list[str]
    professional_roles: list[int] = field(default_factory=list)
    mix_of: list[str] | None = None


class AppConfig(Struct):
    """Application configuration loaded from specialties.yml."""

    specialties: dict[str, SpecialtyDefinition]
    database_path: str = "data/hh_competency.db"
    hh_api_base_url: str = "https://api.hh.ru"
    hh_api_user_agent: str = "hh-competency/0.1.0"
    hh_api_requests_per_second: float = 2.0
    hh_api_max_retries: int = 3
    hh_api_retry_backoff: float = 2.0


# -- Scrape types --


class ScrapeRun(Struct, kw_only=True):
    """Tracks a single scraping session."""

    run_id: str
    specialty: str
    query_params: dict[str, Any] = field(default_factory=dict)
    started_at: datetime
    completed_at: datetime | None = None
    vacancies_found: int = 0
    vacancies_fetched: int = 0


class VacancyData(Struct, kw_only=True):
    """Normalized representation of an HH.ru vacancy."""

    id: str
    name: str
    description: str | None = None
    key_skills: list[dict[str, Any]] = field(default_factory=list)
    salary: dict[str, Any] | None = None
    experience: dict[str, Any] | None = None
    area: dict[str, Any] | None = None
    professional_roles: list[dict[str, Any]] = field(default_factory=list)
    employer: dict[str, Any] | None = None
    published_at: str | None = None

    @classmethod
    def from_api_response(cls, raw: dict[str, Any]) -> VacancyData:
        """Create VacancyData from HH.ru API /vacancies/{id} response."""
        return cls(
            id=str(raw["id"]),
            name=raw["name"],
            description=raw.get("description"),
            key_skills=raw.get("key_skills", []),
            salary=raw.get("salary"),
            experience=raw.get("experience"),
            area=raw.get("area"),
            professional_roles=raw.get("professional_roles", []),
            employer=raw.get("employer"),
            published_at=raw.get("published_at"),
        )


# -- Analysis result types --


class SkillFrequency(Struct):
    """Frequency of a skill lemma within a specialty."""

    specialty: str
    lemma: str
    pos: str
    frequency: int
    vacancy_count: int = 0


class RoleProfile(Struct):
    """A discovered competency-role profile."""

    specialty: str
    role_name: str
    defining_skills: list[tuple[str, float]]
    supporting_skills: list[tuple[str, float]]
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# -- Competency matrix types --


class CompetencyAxis(Struct):
    """A competency dimension discovered from clustering."""

    name: str
    typical_skills: list[str]
    proficiency_levels: int = 5


class CompetencyMatrix(Struct):
    """Multi-axis competency matrix: role level × competency axis → proficiency."""

    specialty: str
    axes: list[CompetencyAxis]
    role_levels: list[str]  # e.g. ["мнс", "нс", "снс", "внс", "гнс", "руководитель"]
    matrix: list[list[int]]  # rows=roles, cols=axes, values=proficiency 0-5
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# -- Role archetype types --


class RoleArchetype(Struct):
    """ISCB-inspired role archetype classification for a role profile.

    Each role cluster is classified into one of six archetypes based
    on the skill category distribution of its defining skills:
    Техник, Исследователь, Методолог, Инженер, Ведущий специалист, Гибрид.
    """

    role_name: str
    archetype_name: str  # "Техник" | "Исследователь" | ...
    description: str
    career_path: str
    competency_emphasis: list[str]


# -- Simulation types --


class SimulationResult(Struct):
    """Result of a Monte Carlo or optimization simulation."""

    scenario_name: str
    specialty: str
    n_samples: int
    mean_coverage: float
    ci_lower: float
    ci_upper: float
    skill_set: list[str]
    match_distribution: list[float] = field(default_factory=list)
