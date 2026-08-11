"""Configuration loader for KRM pipeline.

Reads config.yaml and provides typed access to all pipeline parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Typed wrapper around config.yaml."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = Path(__file__).parent.parent.parent / "config.yaml"
        self._path = Path(path)
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
        return Path(self._data["pipeline"]["output_dir"])

    @property
    def models_dir(self) -> Path:
        return Path(self._data["pipeline"]["models_dir"])

    @property
    def reports_dir(self) -> Path:
        return Path(self._data["pipeline"]["reports_dir"])

    @property
    def data_dir(self) -> Path:
        return self.output_dir

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
    def axis_scores_path(self) -> Path:
        return self.output_dir / "axis_scores.parquet"

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
    def esco_path(self) -> Path:
        return Path(self._data["skills"]["esco_path"])

    @property
    def fuzzy_threshold(self) -> float:
        return self._data["skills"]["fuzzy_threshold"]

    @property
    def tfidf_max_features(self) -> int:
        return self._data["skills"]["tfidf_max_features"]

    # -- Axes ------------------------------------------------------------------

    @property
    def axes(self) -> list[dict[str, str]]:
        return self._data["axes"]

    @property
    def axis_ids(self) -> list[str]:
        return [a["id"] for a in self.axes]

    @property
    def axis_labels_ru(self) -> list[str]:
        return [a["label_ru"] for a in self.axes]

    @property
    def axis_hypotheses(self) -> list[str]:
        return [a["hypothesis"] for a in self.axes]

    @property
    def n_axes(self) -> int:
        return len(self.axes)

    # -- Characteristics -------------------------------------------------------

    @property
    def characteristics_model(self) -> str:
        return self._data["characteristics"]["model"]

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
    def characteristics_hypotheses(self) -> list[dict[str, str]]:
        return self._data["characteristics"]["hypotheses"]

    @property
    def characteristics_path(self) -> Path:
        return self.output_dir / "characteristics.parquet"

    @property
    def experience_patterns(self) -> list[str]:
        return self._data["experience"]["patterns"]

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
        for d in [self.raw_dir, self.output_dir, self.models_dir, self.reports_dir, self.esco_path.parent]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config({self._path})"
