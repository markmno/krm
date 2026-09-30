"""Tests for phase0 configuration properties."""

from __future__ import annotations

from krm.config import Config


class TestPhase0Config:
    """Verify all phase0 config properties load correctly from config.yaml."""

    @classmethod
    def setup_class(cls) -> None:
        cls.config = Config()

    def test_phase0_cdx_rate_limit_rps(self) -> None:
        assert self.config.phase0_cdx_rate_limit_rps == 1

    def test_phase0_cdx_max_per_query(self) -> None:
        assert self.config.phase0_cdx_max_per_query == 150000

    def test_phase0_cdx_endpoint(self) -> None:
        assert "web.archive.org/cdx" in self.config.phase0_cdx_endpoint

    def test_phase0_output_dir(self) -> None:
        assert self.config.phase0_output_dir == "data/phase0"

    def test_phase0_hhru_wayback_enabled(self) -> None:
        assert self.config.phase0_hhru_wayback_enabled is True

    def test_phase0_hhru_wayback_years(self) -> None:
        assert self.config.phase0_hhru_wayback_years == [2010, 2025]

    def test_phase0_hhru_wayback_stem_keywords_has_fizik(self) -> None:
        assert "физик" in self.config.phase0_hhru_wayback_stem_keywords

    def test_phase0_linkedin_wayback_enabled(self) -> None:
        assert self.config.phase0_linkedin_wayback_enabled is True

    def test_phase0_linkedin_wayback_languages(self) -> None:
        assert self.config.phase0_linkedin_wayback_languages == ["en"]

    def test_phase0_linkedin_wayback_stem_keywords_has_scientist(self) -> None:
        assert "scientist" in self.config.phase0_linkedin_wayback_stem_keywords

    def test_phase0_trudvsem_enabled(self) -> None:
        assert self.config.phase0_trudvsem_enabled is True

    def test_phase0_trudvsem_base_url(self) -> None:
        assert self.config.phase0_trudvsem_base_url == "https://opendata.trudvsem.ru/api/v1"

    def test_phase0_trudvsem_date_from(self) -> None:
        assert self.config.phase0_trudvsem_date_from == "2017-01-01"

    def test_phase0_trudvsem_date_to(self) -> None:
        assert self.config.phase0_trudvsem_date_to == "2025-12-31"

    def test_phase0_trudvsem_rate_limit_rps(self) -> None:
        assert self.config.phase0_trudvsem_rate_limit_rps == 1

    def test_phase0_rostud_enabled(self) -> None:
        assert self.config.phase0_rostud_enabled is False

    def test_phase0_rostud_dataset_path(self) -> None:
        assert self.config.phase0_rostud_dataset_path == "data/phase0/rostud_raw"

    def test_phase0_db_path(self) -> None:
        assert self.config.phase0_db_path == "data/phase0/historical.duckdb"

    def test_phase0_url_patterns_structure(self) -> None:
        patterns = self.config.phase0_hhru_wayback_url_patterns
        assert patterns["legacy"] == "hh.ru/vacancy*.do"
        assert patterns["modern"] == "hh.ru/vacancy/[0-9]*"
