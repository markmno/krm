"""Configuration loader for KRM pipeline.

Reads config.yaml and provides typed access to all pipeline parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Typed wrapper around config.yaml."""

    def __init__(
        self,
        path: Path | str | None = None,
        domain: str | None = None,
    ) -> None:
        if path is None:
            path = Path(__file__).parent.parent.parent / "config.yaml"
        self._path = Path(path)
        self._domain = domain
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            msg = f"Config file not found: {self._path}"
            raise FileNotFoundError(msg)
        with open(self._path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    # -- Pipeline paths --------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        return Path(self._data["pipeline"]["raw_dir"])

    @property
    def output_dir(self) -> Path:
        base = Path(self._data["pipeline"]["output_dir"])
        return base / self._domain if self._domain else base

    @property
    def models_dir(self) -> Path:
        base = Path(self._data["pipeline"]["models_dir"])
        return base / self._domain if self._domain else base

    @property
    def reports_dir(self) -> Path:
        base = Path(self._data["pipeline"]["reports_dir"])
        return base / self._domain if self._domain else base

    @property
    def db_path(self) -> Path:
        if self._domain:
            return Path(self._data["domains"][self._domain]["db_path"])
        return Path("data/krm.duckdb")

    @property
    def data_dir(self) -> Path:
        return self.output_dir

    # -- Phase 0: Historical Data Collection ------------------------------------

    @property
    def phase0_output_dir(self) -> str:
        return self._data["phase0"]["output_dir"]

    @property
    def phase0_db_path(self) -> str:
        return self._data["phase0"]["db_path"]

    @property
    def phase0_census_cache_dir(self) -> str:
        return self._data["phase0"]["census_cache_dir"]

    @property
    def phase0_cdx_rate_limit_rps(self) -> int:
        return self._data["phase0"]["cdx"]["rate_limit_rps"]

    @property
    def phase0_cdx_max_per_query(self) -> int:
        return self._data["phase0"]["cdx"]["max_per_query"]

    @property
    def phase0_cdx_endpoint(self) -> str:
        return self._data["phase0"]["cdx"]["endpoint"]

    @property
    def phase0_hhru_wayback_enabled(self) -> bool:
        return self._data["phase0"]["hhru_wayback"]["enabled"]

    @property
    def phase0_hhru_wayback_url_patterns(self) -> dict[str, str]:
        return self._data["phase0"]["hhru_wayback"]["url_patterns"]

    @property
    def phase0_hhru_wayback_years(self) -> list[int]:
        return self._data["phase0"]["hhru_wayback"]["years"]

    @property
    def phase0_hhru_wayback_stem_keywords(self) -> list[str]:
        return self._data["phase0"]["hhru_wayback"]["stem_keywords"]

    @property
    def phase0_linkedin_wayback_enabled(self) -> bool:
        return self._data["phase0"]["linkedin_wayback"]["enabled"]

    @property
    def phase0_linkedin_wayback_url_patterns(self) -> dict[str, str]:
        return self._data["phase0"]["linkedin_wayback"]["url_patterns"]

    @property
    def phase0_linkedin_wayback_languages(self) -> list[str]:
        return self._data["phase0"]["linkedin_wayback"]["languages"]

    @property
    def phase0_linkedin_wayback_stem_keywords(self) -> list[str]:
        return self._data["phase0"]["linkedin_wayback"]["stem_keywords"]

    @property
    def phase0_trudvsem_enabled(self) -> bool:
        return self._data["phase0"]["trudvsem"]["enabled"]

    @property
    def phase0_trudvsem_base_url(self) -> str:
        return self._data["phase0"]["trudvsem"]["base_url"]

    @property
    def phase0_trudvsem_date_from(self) -> str:
        return self._data["phase0"]["trudvsem"]["date_from"]

    @property
    def phase0_trudvsem_date_to(self) -> str:
        return self._data["phase0"]["trudvsem"]["date_to"]

    @property
    def phase0_trudvsem_rate_limit_rps(self) -> int:
        return self._data["phase0"]["trudvsem"]["rate_limit_rps"]

    @property
    def phase0_rostud_enabled(self) -> bool:
        return self._data["phase0"]["rostud"]["enabled"]

    @property
    def phase0_rostud_dataset_path(self) -> str:
        return self._data["phase0"]["rostud"]["dataset_path"]

    @property
    def phase0_telegram_enabled(self) -> bool:
        return self._data["phase0"]["telegram"]["enabled"]

    @property
    def phase0_telegram_api_id(self) -> int:
        return self._data["phase0"]["telegram"]["api_id"]

    @property
    def phase0_telegram_api_hash(self) -> str:
        return self._data["phase0"]["telegram"]["api_hash"]

    @property
    def phase0_telegram_session_name(self) -> str:
        return self._data["phase0"]["telegram"]["session_name"]

    @property
    def phase0_telegram_channels(self) -> list[str]:
        return self._data["phase0"]["telegram"]["channels"]

    @property
    def phase0_telegram_keywords(self) -> list[str]:
        return self._data["phase0"]["telegram"]["keywords"]

    @property
    def phase0_telegram_limit_per_channel(self) -> int:
        return self._data["phase0"]["telegram"]["limit_per_channel"]

    # -- Derived data paths -----------------------------------------------------

    @property
    def classified_path(self) -> Path:
        return self.output_dir / "classified.parquet"

    @property
    def roles_path(self) -> Path:
        return self.output_dir / "roles.parquet"

    @property
    def skills_per_role_path(self) -> Path:
        return self.output_dir / "skills_per_role.parquet"

    @property
    def characteristic_scores_path(self) -> Path:
        return self.output_dir / "characteristic_scores.parquet"

    @property
    def raw_vacancies_path(self) -> Path:
        return self.output_dir / "raw_vacancies.parquet"

    # -- Collection ------------------------------------------------------------

    @property
    def date_from(self) -> str:
        return self._data["collection"]["date_from"]

    @property
    def keywords(self) -> list[str]:
        return self._data["collection"]["keywords"]

    @property
    def professional_roles(self) -> list[int]:
        return self._data["collection"]["professional_roles"]

    @property
    def categories(self) -> list[int]:
        return self._data["collection"]["categories"]

    @property
    def exclude_roles(self) -> list[int]:
        return self._data["collection"]["exclude_roles"]

    @property
    def rate_limit_rps(self) -> int:
        return self._data["collection"]["rate_limit_rps"]

    @property
    def http_proxy(self) -> str:
        return self._data["collection"].get("http_proxy", "")

    @property
    def exclude_terms(self) -> list[str]:
        return self._data["collection"].get("exclude_terms", [])

    # -- Domain-specific collection ---------------------------------------------

    @property
    def domains(self) -> dict[str, dict[str, Any]]:
        return self._data.get("domains", {})

    def domain_db_path(self, domain: str) -> Path:
        return Path(self._data["domains"][domain]["db_path"])

    def domain_keywords(self, domain: str) -> list[str]:
        return self._data["domains"][domain]["keywords"]

    def domain_name(self, domain: str) -> str:
        return self._data["domains"][domain].get("name", domain)

    # -- Classification --------------------------------------------------------

    @property
    def classification_model(self) -> str:
        return self._data["classification"]["model"]

    @property
    def stem_threshold(self) -> float:
        return self._data["classification"]["stem_threshold"]

    @property
    def it_threshold(self) -> float:
        return self._data["classification"]["it_threshold"]

    @property
    def interdisciplinary_threshold(self) -> float:
        return self._data["classification"]["interdisciplinary_threshold"]

    # -- Roles (clustering) ----------------------------------------------------

    @property
    def embedding_model(self) -> str:
        return self._data["roles"]["embedding_model"]

    @property
    def umap_n_components(self) -> int:
        return self._data["roles"]["umap_n_components"]

    @property
    def hdbscan_min_cluster_size(self) -> int:
        return self._data["roles"]["hdbscan_min_cluster_size"]

    @property
    def hdbscan_cluster_selection_epsilon(self) -> float:
        return self._data["roles"]["hdbscan_cluster_selection_epsilon"]

    # -- Skills ----------------------------------------------------------------

    @property
    def tfidf_max_features(self) -> int:
        return self._data["skills"]["tfidf_max_features"]

    # -- Skill Extraction (data-driven, Phase 2b + 4) --------------------------

    @property
    def skill_extraction_min_phrase_length(self) -> int:
        return self._data["skill_extraction"]["min_phrase_length"]

    @property
    def skill_extraction_max_phrase_length(self) -> int:
        return self._data["skill_extraction"]["max_phrase_length"]

    @property
    def skill_extraction_similarity_threshold(self) -> float:
        return self._data["skill_extraction"]["similarity_threshold"]

    @property
    def skill_extraction_min_doc_frequency(self) -> int:
        return self._data["skill_extraction"]["min_doc_frequency"]

    @property
    def skill_extraction_max_skills_per_vacancy(self) -> int:
        return self._data["skill_extraction"]["max_skills_per_vacancy"]

    @property
    def skill_extraction_embedding_model(self) -> str:
        return self._data["skill_extraction"]["embedding_model"]

    # -- Characteristics (unified — Phase 2.5 + Phase 5) -------------------------

    @property
    def characteristic_ids(self) -> list[str]:
        return [h["id"] for h in self._data["characteristics"]["hypotheses"]]

    @property
    def characteristic_hypotheses(self) -> dict[str, str]:
        return {h["id"]: h["hypothesis"] for h in self._data["characteristics"]["hypotheses"]}

    @property
    def characteristic_labels_ru(self) -> dict[str, str]:
        return {h["id"]: h["label_ru"] for h in self._data["characteristics"]["hypotheses"]}

    @property
    def characteristic_labels_en(self) -> dict[str, str]:
        return {h["id"]: h["label_en"] for h in self._data["characteristics"]["hypotheses"]}

    @property
    def characteristic_skill_triggers(self) -> dict[str, list[str]]:
        return {
            h["id"]: h.get("skill_category_triggers", [])
            for h in self._data["characteristics"]["hypotheses"]
        }

    @property
    def characteristic_seed_glossaries(self) -> dict[str, list[str]]:
        return {
            h["id"]: h.get("seed_glossaries", [])
            for h in self._data["characteristics"]["hypotheses"]
        }

    @property
    def n_characteristics(self) -> int:
        return len(self._data["characteristics"]["hypotheses"])

    @property
    def characteristics_model(self) -> str:
        # Unified with Phase 2 classification: one NLI model for the pipeline.
        return self.classification_model

    @property
    def characteristics_device(self) -> str:
        return self._data["characteristics"]["device"]

    @property
    def characteristics_batch_size(self) -> int:
        return self._data["characteristics"]["batch_size"]

    @property
    def characteristics_confidence_threshold(self) -> float:
        return self._data["characteristics"]["confidence_threshold"]

    @property
    def characteristics_path(self) -> Path:
        return self.output_dir / "characteristics.parquet"

    @property
    def experience_patterns(self) -> list[str]:
        return self._data["experience"]["patterns"]

    @property
    def experience_bins(self) -> list[float]:
        return [float(b) for b in self._data["experience"].get("bins", [0, 1, 3, 5, 10])]

    # -- Soft competences (Phase 5b) -------------------------------------------

    @property
    def soft_competence_ids(self) -> list[str]:
        return [h["id"] for h in self._data["soft_competences"]["hypotheses"]]

    @property
    def soft_competence_hypotheses(self) -> dict[str, str]:
        return {h["id"]: h["hypothesis"] for h in self._data["soft_competences"]["hypotheses"]}

    @property
    def soft_competence_labels_ru(self) -> dict[str, str]:
        return {h["id"]: h["label_ru"] for h in self._data["soft_competences"]["hypotheses"]}

    @property
    def soft_competence_model(self) -> str:
        return self._data["soft_competences"]["model"]

    @property
    def soft_competence_device(self) -> str:
        return self._data["soft_competences"]["device"]

    @property
    def soft_competence_batch_size(self) -> int:
        return self._data["soft_competences"].get("batch_size", 32)

    @property
    def soft_competence_confidence_threshold(self) -> float:
        return self._data["soft_competences"].get("confidence_threshold", 0.3)

    @property
    def soft_competence_min_descriptions(self) -> int:
        return self._data["soft_competences"].get("min_descriptions", 5)

    @property
    def soft_baseline(self) -> dict[str, dict[str, float]]:
        return self._data["soft_competences"]["baseline"]

    @property
    def soft_rubric(self) -> dict[str, dict[str, float]]:
        return self._data["soft_competences"].get("soft_rubric", {})

    @property
    def archetype_names(self) -> list[str]:
        return self._data["soft_competences"]["archetypes"]

    @property
    def skills_top_n(self) -> int:
        return self._data.get("skills_display", {}).get("top_n", 15)

    @property
    def vacancy_roles_path(self) -> Path:
        return self.output_dir / "vacancy_roles.parquet"

    @property
    def vacancy_experience_path(self) -> Path:
        return self.output_dir / "vacancy_experience.parquet"

    @property
    def skill_characteristic_scores_path(self) -> Path:
        return self.output_dir / "skill_characteristic_scores.parquet"

    @property
    def soft_scores_path(self) -> Path:
        return self.output_dir / "soft_scores.parquet"

    @property
    def role_archetypes_path(self) -> Path:
        return self.output_dir / "role_archetypes.parquet"

    # -- Testing ---------------------------------------------------------------

    @property
    def classification_sample_size(self) -> int:
        return self._data["testing"]["classification_sample_size"]

    @property
    def skill_precision_at_k(self) -> int:
        return self._data["testing"]["skill_precision_at_k"]

    @property
    def expert_validation_roles(self) -> int:
        return self._data["testing"]["expert_validation_roles"]

    @property
    def bootstrap_iterations(self) -> int:
        return self._data["testing"]["bootstrap_iterations"]

    # -- Helper ----------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [self.raw_dir, self.output_dir, self.models_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config({self._path})"
