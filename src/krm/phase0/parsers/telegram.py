"""Deterministic parser for Telegram vacancy-channel messages.

Turns free-text Russian/English vacancy messages posted to Telegram channels
into a canonical dict matching the hh.ru API field names. Pure and tolerant:
no LLM, no network, never raises — missing markers fall back to best-effort
defaults.
"""

from __future__ import annotations

from krm.phase0.parsers.common import parse_iso_date, parse_salary_from_text

#: Markers that introduce the job title.  Hashtag markers are usually a
#: standalone line followed by the title on the next line; label markers carry
#: the title inline after the colon.
_NAME_HASHTAG_MARKERS: tuple[str, ...] = (
    "#вакансия",
    "#vacancy",
    "#вакансии",
    "#vacancies",
)
_NAME_LABEL_MARKERS: tuple[str, ...] = (
    "вакансия:",
    "должность:",
    "position:",
)

_EMPLOYER_MARKERS: tuple[str, ...] = (
    "компания:",
    "организация:",
    "работодатель:",
    "company:",
    "employer:",
)

_AREA_MARKERS: tuple[str, ...] = (
    "город:",
    "локация:",
    "расположение:",
    "location:",
)


def _first_non_empty_line(text: str) -> str | None:
    """Return the first non-whitespace line, or None."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _label_value(line: str, marker: str) -> str | None:
    """Return the text after a ``Label:`` marker on a line, or None."""
    stripped = line.strip()
    if not stripped.lower().startswith(marker):
        return None
    rest = stripped[len(marker) :].strip(" \t:-–—")
    return rest or None


def _extract_name(text: str) -> str | None:
    """Extract the job title from name markers, else the first non-empty line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        low = stripped.lower()

        for marker in _NAME_LABEL_MARKERS:
            value = _label_value(stripped, marker)
            if value:
                return value

        for marker in _NAME_HASHTAG_MARKERS:
            if low == marker:
                # Standalone hashtag — title is on a following line. If that
                # line is itself a label, take the label's value instead.
                for nxt in lines[i + 1 :]:
                    if not nxt.strip():
                        continue
                    for lbl in _NAME_LABEL_MARKERS:
                        value = _label_value(nxt, lbl)
                        if value:
                            return value
                    return nxt.strip()
            elif low.startswith(marker + " "):
                rest = stripped[len(marker) :].strip()
                if rest:
                    return rest

    return _first_non_empty_line(text)


def _extract_label(text: str, markers: tuple[str, ...]) -> str | None:
    """Return the text after the first matching ``Label:`` marker, or None."""
    for line in text.splitlines():
        for marker in markers:
            value = _label_value(line, marker)
            if value:
                return value
    return None


def parse_telegram_message(
    text: str,
    *,
    channel: str,
    message_id: int,
    date: str | None = None,
) -> dict[str, object]:
    """Parse a Telegram vacancy message into a canonical hh.ru-shaped dict.

    Best-effort and tolerant: extracts the job title, employer, area and
    salary from Russian/English markers, falling back to the first line for
    the title. Never raises.

    Args:
        text: Raw message text.
        channel: Telegram channel username (used to build ``_source_url``).
        message_id: Telegram message id (used to build ``_source_url``).
        date: Optional publication date string (parsed via ``parse_iso_date``).

    Returns:
        Dict with keys ``name``, ``description``, ``employer``, ``area``,
        ``salary``, ``published_at``, ``key_skills``, ``experience``,
        ``professional_roles`` and ``_source_url``.
    """
    employer_name = _extract_label(text, _EMPLOYER_MARKERS)
    area_name = _extract_label(text, _AREA_MARKERS)

    return {
        "name": _extract_name(text) or "",
        "description": text.strip(),
        "employer": {"name": employer_name},
        "area": {"name": area_name},
        "salary": parse_salary_from_text(text),
        "published_at": parse_iso_date(date) if date else None,
        "key_skills": [],
        "experience": {"name": None},
        "professional_roles": [],
        "_source_url": f"https://t.me/{channel}/{message_id}",
    }
