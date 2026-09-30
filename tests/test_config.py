"""Tests for the unified config layer: 7 hard axes + soft competences + display params."""

from __future__ import annotations

from krm.config import Config


class TestConfig:
    """Verify the 7-axis characteristics and soft-competence config properties."""

    @classmethod
    def setup_class(cls) -> None:
        cls.config = Config()

    def test_characteristic_ids_order(self) -> None:
        assert self.config.characteristic_ids == [
            "domain_knowledge",
            "experimental",
            "data_analysis",
            "computational",
            "professional_texts",
            "t_profile",
            "management",
        ]

    def test_soft_competence_ids(self) -> None:
        assert self.config.soft_competence_ids == [
            "thinking",
            "teamwork",
            "leadership",
            "professional_culture",
        ]

    def test_soft_baseline_lookup(self) -> None:
        assert self.config.soft_baseline["Техник"]["leadership"] == 1

    def test_experience_bins(self) -> None:
        assert self.config.experience_bins == [0, 1, 3, 5, 10]

    def test_skills_top_n(self) -> None:
        assert self.config.skills_top_n == 15

    def test_n_characteristics(self) -> None:
        assert self.config.n_characteristics == 7
