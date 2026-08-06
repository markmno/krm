"""Configuration loader for hh-competency.

Loads specialties.yml and produces a typed AppConfig.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hh_competency.storage.models import AppConfig, SpecialtyDefinition


class ConfigError(Exception):
    """Configuration validation error."""


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate application configuration from YAML.

    Args:
        config_path: Path to specialties.yml or similar config file.

    Returns:
        Typed AppConfig with all specialties and settings.

    Raises:
        ConfigError: If config is invalid (missing required fields, invalid refs).
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw or "specialties" not in raw:
        raise ConfigError("Config must contain 'specialties' key")

    specialties: dict[str, SpecialtyDefinition] = {}
    for name, spec in raw["specialties"].items():
        if "search_keywords" not in spec:
            raise ConfigError(
                f"Specialty '{name}' missing required field: search_keywords"
            )
        specialties[name] = SpecialtyDefinition(
            name=name,
            search_keywords=spec["search_keywords"],
            nlp_keywords=spec.get("nlp_keywords", []),
            professional_roles=spec.get("professional_roles", []),
            mix_of=spec.get("mix_of"),
        )

    # Validate mix_of references
    for name, spec in specialties.items():
        if spec.mix_of:
            for parent in spec.mix_of:
                if parent not in specialties:
                    raise ConfigError(
                        f"Specialty '{name}' mix_of references unknown "
                        f"specialty '{parent}'"
                    )

    db_config = raw.get("database", {})
    api_config = raw.get("hh_api", {})

    return AppConfig(
        specialties=specialties,
        database_path=db_config.get("path", "data/hh_competency.db"),
        hh_api_base_url=api_config.get("base_url", "https://api.hh.ru"),
        hh_api_user_agent=api_config.get("user_agent", "hh-competency/0.1.0"),
        hh_api_requests_per_second=api_config.get("requests_per_second", 2.0),
        hh_api_max_retries=api_config.get("max_retries", 3),
        hh_api_retry_backoff=api_config.get("retry_backoff", 2.0),
    )
