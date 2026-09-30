"""Phase 0 schema mapper — normalizes extracted data from all 4 sources
(Wayback HH.ru, Wayback LinkedIn, Trudvsem API, Rostrud CSV) into the
hh.ru API JSON schema with Phase 0 metadata.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Schema constants — target hh.ru API JSON shape
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"id", "name"}
OPTIONAL_FIELDS = {
    "description",
    "employer",
    "area",
    "salary",
    "published_at",
    "experience",
    "key_skills",
    "professional_roles",
    "schedule",
    "employment",
    "archived",
}
PHASE0_META_FIELDS = {"_phase0_source", "_phase0_capture_ts", "_phase0_original_url"}

SOURCE_TYPES = (
    "wayback-hhru",
    "wayback-linkedin",
    "trudvsem",
    "rostud",
    "dano-hse-2023",
    "telegram",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LINKEDIN_JOB_ID_RE = re.compile(r"/jobs2?/view/(?:[^/]+-)?(\d+)")

_TELEGRAM_URL_RE = re.compile(r"https?://t\.me/([^/]+)/(\d+)")


def _extract_linkedin_job_id(url: str) -> str | None:
    """Extract numeric LinkedIn job ID from a URL like
    /jobs2/view/12345 or /jobs/view/slug-1234567890.
    """
    match = _LINKEDIN_JOB_ID_RE.search(url)
    if match:
        return match.group(1)
    return None


def _extract_hhru_vacancy_id(url: str) -> str | None:
    """Extract numeric HH.ru vacancy ID from a URL like
    /vacancy/12345678 or /vacancy/12345678?query=...
    """
    match = re.search(r"/vacancy/(\d+)", url)
    if match:
        return match.group(1)
    return None


def _ensure_key_skills(raw: list[Any] | None) -> list[dict[str, str]]:
    """Coerce a raw value into the canonical [{"name": str}] format."""
    if raw is None:
        return []
    result: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict) and "name" in item:
            result.append({"name": str(item["name"])})
        elif isinstance(item, str):
            result.append({"name": item})
    return result


def _ensure_professional_roles(raw: list[Any] | None) -> list[dict[str, Any]]:
    """Coerce a raw value into [{"id": int, "name": str}] format."""
    if raw is None:
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            role_id = int(item.get("id", 0))
            role_name = str(item.get("name", ""))
            if role_id and role_name:
                result.append({"id": role_id, "name": role_name})
    return result


# ---------------------------------------------------------------------------
# Source-specific mappers
# ---------------------------------------------------------------------------


def map_hhru_wayback(raw: dict[str, Any], source_url: str, capture_ts: str) -> dict[str, Any]:
    """Map a Wayback Machine HH.ru page-extracted dict to the hh.ru schema.

    HH.ru extraction already uses hh.ru API field names, so this is
    mostly a direct pass-through with metadata attachment.
    """
    vacancy_id = raw.get("id")
    if not vacancy_id:
        extracted = _extract_hhru_vacancy_id(source_url)
        vacancy_id = extracted if extracted else source_url

    mapped: dict[str, Any] = {
        "id": str(vacancy_id),
        "name": raw.get("name", ""),
        "_phase0_source": "wayback-hhru",
        "_phase0_capture_ts": capture_ts,
        "_phase0_original_url": source_url,
    }

    # Direct pass-through of optional fields when present
    for field in OPTIONAL_FIELDS:
        if field in raw:
            mapped[field] = raw[field]

    # Coerce structured fields to canonical shape
    mapped["key_skills"] = _ensure_key_skills(raw.get("key_skills"))
    mapped["professional_roles"] = _ensure_professional_roles(raw.get("professional_roles"))

    return mapped


def map_linkedin_wayback(raw: dict[str, Any], source_url: str, capture_ts: str) -> dict[str, Any]:
    """Map a Wayback Machine LinkedIn page-extracted dict to the hh.ru schema.

    LinkedIn extraction uses LinkedIn-native field names; this mapper
    translates them and attaches Phase 0 metadata.
    """
    # ── id ────────────────────────────────────────────────────────────
    linkedin_id = raw.get("id")
    if not linkedin_id:
        numeric = _extract_linkedin_job_id(source_url)
        linkedin_id = numeric if numeric else linkedin_id
    vacancy_id = f"linkedin-{linkedin_id}" if linkedin_id else source_url

    mapped: dict[str, Any] = {
        "id": str(vacancy_id),
        "name": raw.get("name", ""),
        "_phase0_source": "wayback-linkedin",
        "_phase0_capture_ts": capture_ts,
        "_phase0_original_url": source_url,
    }

    # ── employer ──────────────────────────────────────────────────────
    employer_raw = raw.get("employer")
    if isinstance(employer_raw, dict) and "name" in employer_raw:
        mapped["employer"] = {"name": employer_raw["name"]}
    elif isinstance(employer_raw, str):
        mapped["employer"] = {"name": employer_raw}

    # ── area ──────────────────────────────────────────────────────────
    area_raw = raw.get("area")
    if isinstance(area_raw, dict) and "name" in area_raw:
        mapped["area"] = {"name": area_raw["name"]}
    elif isinstance(area_raw, str):
        mapped["area"] = {"name": area_raw}

    # ── description (with optional industry prefix) ───────────────────
    description = raw.get("description", "")
    industry = raw.get("industry")
    if industry and description:
        mapped["description"] = f"[Industry: {industry}] {description}"
    elif description:
        mapped["description"] = description

    # ── key_skills ────────────────────────────────────────────────────
    mapped["key_skills"] = _ensure_key_skills(raw.get("key_skills"))

    return mapped


def map_trudvsem(raw: dict[str, Any], capture_ts: str) -> dict[str, Any]:
    """Map a Trudvsem API response dict to the hh.ru schema.

    Trudvsem uses Russian field names (``job-name``, ``requirements``,
    ``duty``, ``vac_url``, ``creation-date``, ``skills``); this mapper
    translates them to standard hh.ru API field names.

    The caller is responsible for unwrapping the ``{"vacancy": {...}}``
    API response shape before invoking this function.
    """
    # ── id ────────────────────────────────────────────────────────────
    raw_id = raw.get("id") or raw.get("vacancy_id") or ""
    vacancy_id = f"trudvsem-{raw_id}" if raw_id else ""

    # ── name — real API uses ``job-name``, fall back to legacy aliases
    name = (
        raw.get("job-name", "")
        or raw.get("vacancy_name", "")
        or raw.get("name", "")
    )

    # ── original URL — real API uses ``vac_url``
    original_url = (
        raw.get("vac_url", "")
        or raw.get("url", "")
        or raw.get("source_url", "")
    )

    mapped: dict[str, Any] = {
        "id": str(vacancy_id),
        "name": name,
        "_phase0_source": "trudvsem",
        "_phase0_capture_ts": capture_ts,
        "_phase0_original_url": original_url,
    }

    # ── description — combine requirements + duty for full context
    requirements = raw.get("requirements", "")
    duty = raw.get("duty", "")
    legacy_desc = raw.get("vacancy_description") or raw.get("description")
    if legacy_desc:
        # Old format still has a single description
        mapped["description"] = legacy_desc
    elif requirements or duty:
        parts: list[str] = []
        if requirements:
            parts.append(f"Требования: {requirements}")
        if duty:
            parts.append(f"Обязанности: {duty}")
        mapped["description"] = "\n".join(parts)

    # ── employer ──────────────────────────────────────────────────────
    company = raw.get("company")
    company_name = raw.get("company_name")
    if isinstance(company, dict) and "name" in company:
        mapped["employer"] = {"name": company["name"]}
    elif company_name:
        mapped["employer"] = {"name": company_name}

    # ── area ──────────────────────────────────────────────────────────
    region = raw.get("region")
    region_name = raw.get("region_name")
    if isinstance(region, dict) and "name" in region:
        mapped["area"] = {"name": region["name"]}
    elif region_name:
        mapped["area"] = {"name": region_name}

    # ── salary ────────────────────────────────────────────────────────
    salary_from = raw.get("salary_min")
    salary_to = raw.get("salary_max")
    if salary_from is not None or salary_to is not None:
        mapped["salary"] = {
            "from": salary_from,
            "to": salary_to,
            "currency": raw.get("salary_currency", "RUR"),
        }

    # ── key_skills — real API uses ``skills`` (array of strings)
    raw_skills = raw.get("skills") or raw.get("key_skills")
    mapped["key_skills"] = _ensure_key_skills(raw_skills)

    # ── published_at — real API uses ``creation-date``
    creation_date = raw.get("creation-date") or raw.get("creationDate") or raw.get("creation_date")
    if creation_date:
        mapped["published_at"] = str(creation_date)

    # ── professional_roles — use ``category.specialisation`` as a role hint
    category = raw.get("category")
    if isinstance(category, dict) and category.get("specialisation"):
        prof_name = str(category["specialisation"])
        # Map Trudvsem category names to hh.ru professional_role names
        mapped["professional_roles"] = [{"id": 0, "name": prof_name}]
    else:
        mapped["professional_roles"] = []

    return mapped


def map_rostud(raw: dict[str, Any], capture_ts: str) -> dict[str, Any]:
    """Map a Rostrud CSV row dict to the hh.ru schema.

    Rostrud uses Russian column names with multiple possible variants;
    this mapper handles them flexibly.
    """
    # ── id ────────────────────────────────────────────────────────────
    raw_id = raw.get("id")
    if not raw_id:
        # Generate a stable ID from the row content
        name = (
            raw.get("Должность")
            or raw.get("position")
            or raw.get("name")
            or ""
        )
        raw_id = str(abs(hash(f"{name}{raw.get('Работодатель', '')}{capture_ts}")))

    mapped: dict[str, Any] = {
        "id": str(raw_id),
        "name": (
            raw.get("Должность")
            or raw.get("position")
            or raw.get("name")
            or ""
        ),
        "_phase0_source": "rostud",
        "_phase0_capture_ts": capture_ts,
        "_phase0_original_url": raw.get("url", "") or raw.get("source_url", ""),
    }

    # ── description ───────────────────────────────────────────────────
    mapped["description"] = raw.get("Описание") or raw.get("description")

    # ── employer ──────────────────────────────────────────────────────
    employer_name = (
        raw.get("Работодатель")
        or raw.get("employer")
        or raw.get("company")
    )
    if employer_name:
        mapped["employer"] = {"name": employer_name}

    # ── area ──────────────────────────────────────────────────────────
    region_name = (
        raw.get("Регион")
        or raw.get("region")
        or raw.get("location")
    )
    if region_name:
        mapped["area"] = {"name": region_name}

    # ── key_skills ────────────────────────────────────────────────────
    mapped["key_skills"] = _ensure_key_skills(raw.get("key_skills"))

    return mapped


def map_telegram(raw: dict[str, Any], capture_ts: str) -> dict[str, Any]:
    """Map a parsed Telegram message dict to the hh.ru schema.

    The parser produces ``_source_url`` (``https://t.me/{channel}/{message_id}``)
    from which a stable id ``telegram-{channel}-{message_id}`` is derived.
    """
    source_url = raw.get("_source_url", "") or ""

    # ── id ────────────────────────────────────────────────────────────
    url_match = _TELEGRAM_URL_RE.search(source_url)
    if url_match:
        vacancy_id = f"telegram-{url_match.group(1)}-{url_match.group(2)}"
    else:
        vacancy_id = f"telegram-{source_url}"

    mapped: dict[str, Any] = {
        "id": str(vacancy_id),
        "name": raw.get("name", ""),
        "_phase0_source": "telegram",
        "_phase0_capture_ts": capture_ts,
        "_phase0_original_url": source_url,
    }

    # ── employer ──────────────────────────────────────────────────────
    employer_raw = raw.get("employer")
    if isinstance(employer_raw, dict) and employer_raw.get("name"):
        mapped["employer"] = {"name": employer_raw["name"]}
    elif isinstance(employer_raw, str) and employer_raw:
        mapped["employer"] = {"name": employer_raw}

    # ── area ──────────────────────────────────────────────────────────
    area_raw = raw.get("area")
    if isinstance(area_raw, dict) and area_raw.get("name"):
        mapped["area"] = {"name": area_raw["name"]}
    elif isinstance(area_raw, str) and area_raw:
        mapped["area"] = {"name": area_raw}

    # ── description ───────────────────────────────────────────────────
    description = raw.get("description")
    if description:
        mapped["description"] = description

    # ── salary ────────────────────────────────────────────────────────
    salary = raw.get("salary")
    if isinstance(salary, dict):
        mapped["salary"] = {
            "from": salary.get("from"),
            "to": salary.get("to"),
            "currency": salary.get("currency"),
        }

    # ── key_skills ────────────────────────────────────────────────────
    mapped["key_skills"] = _ensure_key_skills(raw.get("key_skills"))

    # ── published_at ──────────────────────────────────────────────────
    published_at = raw.get("published_at")
    if published_at:
        mapped["published_at"] = str(published_at)

    return mapped


# ---------------------------------------------------------------------------
# Normalize & validate
# ---------------------------------------------------------------------------

SOURCE_MAPPERS = {
    "wayback-hhru": map_hhru_wayback,
    "wayback-linkedin": map_linkedin_wayback,
    "trudvsem": map_trudvsem,
    "rostud": map_rostud,
    "telegram": map_telegram,
}


def normalize_record(mapped: dict[str, Any]) -> dict[str, Any]:
    """Fill missing optional fields with None and validate required fields.

    Raises ValueError if a REQUIRED_FIELD is missing.
    """
    for field in REQUIRED_FIELDS:
        if not mapped.get(field):
            raise ValueError(f"Missing required field: {field!r}")

    for field in OPTIONAL_FIELDS:
        mapped.setdefault(field, None)

    for meta in PHASE0_META_FIELDS:
        mapped.setdefault(meta, None)

    # Ensure structured fields use canonical format
    if mapped.get("key_skills") is not None:
        mapped["key_skills"] = _ensure_key_skills(mapped["key_skills"])

    return mapped


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []

    if not record.get("name"):
        errors.append("Missing or empty 'name' field")

    source = record.get("_phase0_source", "")
    if source not in SOURCE_TYPES:
        errors.append(
            f"Invalid _phase0_source {source!r}; must be one of {SOURCE_TYPES!r}"
        )

    if not record.get("_phase0_original_url"):
        errors.append("Missing or empty _phase0_original_url")

    return errors


# ---------------------------------------------------------------------------
# Enrichment — derived education, experience_years, _derived_year
# ---------------------------------------------------------------------------

# Russian education level extraction patterns (order matters: higher first)
_EDUCATION_PATTERNS_RU: list[tuple[str, str]] = [
    (r"(?i)доктор\s+наук", "doctor_of_sciences"),
    (r"(?i)кандидат\s+наук", "candidate_of_sciences"),
    (r"(?i)аспирант|аспирантур", "phd_student"),
    (r"(?i)магистр|магистратур|master", "masters"),
    (r"(?i)бакалавр|bachelor", "bachelors"),
    (r"(?i)высш[а-я]*\s*(?:образован|профессиональн)", "higher_education"),
    (r"(?i)неполное\s*высш", "incomplete_higher"),
    (r"(?i)средн[а-я]*\s*(?:специальн|профессиональн|техническ)", "secondary_vocational"),
    (r"(?i)средн[а-я]*\s*образован", "secondary"),
]

# English education level patterns
_EDUCATION_PATTERNS_EN: list[tuple[str, str]] = [
    (r"(?i)\bPhD\b|doctorate|doctoral", "doctorate"),
    (r"(?i)master'?s?\s*degree|M\.?Sc|M\.?A\.|MBA", "masters"),
    (r"(?i)bachelor'?s?\s*degree|B\.?Sc|B\.?A\.|undergraduate", "bachelors"),
    (r"(?i)associate'?s?\s*degree|A\.?A\.|A\.?S\.", "associates"),
    (r"(?i)high\s*school\s*diploma|GED", "high_school"),
]

# Numeric experience pattern: "1-3 years", "от 3 лет", "3+ years", "3 года"
_EXPERIENCE_YEARS_RE = re.compile(
    r"(?:от\s+|from\s+)?(\d+(?:\.\d+)?)\s*(?:[-–до]+\s*(\d+(?:\.\d+)?)\s*)?"
    r"(?:год|лет|года|years?|yrs?|yr)",
    re.IGNORECASE,
)


def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    """Add derived fields: _derived_year, experience_years, education.

    Call AFTER normalize_record. Modifies record in place and returns it.
    """
    record["_derived_year"] = _extract_year(record.get("published_at")) or _extract_year(
        record.get("_phase0_capture_ts")
    )
    record["experience_years"] = _parse_experience_years(record.get("experience"))
    record["education"] = _extract_education(record)
    return record


def _extract_year(published_at: str | None) -> int | None:
    """Extract integer year from published_at ISO string."""
    if not published_at:
        return None
    m = re.match(r"(\d{4})", str(published_at).strip())
    if m:
        return int(m.group(1))
    return None


def _parse_experience_years(experience: dict[str, Any] | None) -> dict[str, float | None] | None:
    """Parse experience.name text into {min: float | None, max: float | None}.

    Handles: 'От 1 года до 3 лет', '1-3 years', '3+ years', 'Нет опыта', etc.
    """
    if not experience or not isinstance(experience, dict):
        return None
    name = experience.get("name")
    if not name or not isinstance(name, str):
        return None

    name_lower = name.lower().strip()

    if any(w in name_lower for w in ("нет опыта", "без опыта", "no experience", "entry level", "internship")):
        return {"min": 0.0, "max": 0.0}

    m = _EXPERIENCE_YEARS_RE.search(name)
    if m:
        min_val = float(m.group(1))
        max_val = float(m.group(2)) if m.group(2) else None
        return {"min": min_val, "max": max_val}

    return None


def _extract_education(record: dict[str, Any]) -> str | None:
    """Extract education level from description text using regex patterns.

    Checks Russian patterns first (for hh.ru/trudvsem/rostud), then English
    (for LinkedIn). Returns the highest matching level.
    """
    desc = record.get("description")
    if not desc or not isinstance(desc, str):
        return None

    patterns = _EDUCATION_PATTERNS_RU + _EDUCATION_PATTERNS_EN
    for regex, level in patterns:
        if re.search(regex, desc):
            return level

    return None
