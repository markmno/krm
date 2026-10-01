"""Tests for phase_2_5_characteristics.py — NLI default + LLM dispatch.

No network: the LLM path is exercised by monkeypatching
``krm.lib.llm.LLMClient.run_many`` and the NLI path by monkeypatching
``_build_characteristic_classifier`` (which lazy-imports ``transformers``).
"""

from __future__ import annotations

import pytest

from krm.config import Config
from krm.lib.llm import LLMClient
from krm.phase_2_5_characteristics import (
    CharacteristicScore,
    CharacteristicScoresResponse,
    extract_characteristics,
)

_COLUMNS = ["vacancy_id", "characteristic_id", "characteristic_label", "confidence"]


def _enable_llm(tmp_path, monkeypatch) -> Config:
    """Real config with the LLM phase flag flipped on and the cache redirected."""
    config = Config()
    monkeypatch.setattr(Config, "use_llm_phase_2_5", property(lambda self: True))
    monkeypatch.setattr(
        Config, "llm_cache_dir", property(lambda self: str(tmp_path / "llm_cache"))
    )
    return config


def _fake_nli_classifier(texts: list[str], labels: list[str]) -> list[dict]:
    """Entail only the first hypothesis at 0.9, everything else at 0.0."""
    return [
        {"labels": labels, "scores": [0.9 if lab == labels[0] else 0.0 for lab in labels]}
        for _ in texts
    ]


class TestCharacteristicScoreModel:
    def test_confidence_is_clamped_to_unit_interval(self) -> None:
        assert CharacteristicScore(characteristic_id="x", confidence=5.0).confidence == 1.0
        assert CharacteristicScore(characteristic_id="x", confidence=-0.5).confidence == 0.0
        assert (
            CharacteristicScore(characteristic_id="x", confidence=0.42).confidence == 0.42
        )


class TestNliDefault:
    def test_nli_is_default_path(self, monkeypatch) -> None:
        config = Config()  # use_llm_phase_2_5 == False
        monkeypatch.setattr(
            "krm.phase_2_5_characteristics._build_characteristic_classifier",
            lambda _config: _fake_nli_classifier,
        )

        df = extract_characteristics(config, ["лабораторная работа"], ["v1"])

        assert list(df.columns) == _COLUMNS
        assert len(df) == 1
        row = df.iloc[0]
        assert row["vacancy_id"] == "v1"
        assert row["characteristic_id"] == config.characteristic_ids[0]
        assert row["characteristic_label"] == config.characteristic_labels_ru[
            config.characteristic_ids[0]
        ]
        assert row["confidence"] == pytest.approx(0.9)

    def test_nli_empty_description_yields_no_rows(self, monkeypatch) -> None:
        config = Config()
        monkeypatch.setattr(
            "krm.phase_2_5_characteristics._build_characteristic_classifier",
            lambda _config: _fake_nli_classifier,
        )

        df = extract_characteristics(config, [None, ""], ["v1", "v2"])

        assert df.empty


class TestLlmPath:
    def test_llm_returns_shaped_rows_and_drops_unknown_ids(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _enable_llm(tmp_path, monkeypatch)

        resp = CharacteristicScoresResponse(characteristics=[
            CharacteristicScore(characteristic_id="domain_knowledge", confidence=0.9),
            CharacteristicScore(characteristic_id="experimental", confidence=0.1),
            CharacteristicScore(characteristic_id="not_a_real_axis", confidence=0.7),
        ])

        async def fake_run_many(self, prompts, schema) -> list:
            assert schema is CharacteristicScoresResponse
            assert len(prompts) == 1
            assert all(cid in prompts[0].user for cid in config.characteristic_ids)
            return [resp]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)

        df = extract_characteristics(config, ["лабораторная работа"], ["v1"])

        assert list(df.columns) == _COLUMNS
        assert len(df) == 1
        row = df.iloc[0]
        assert row["vacancy_id"] == "v1"
        assert row["characteristic_id"] == "domain_knowledge"
        assert row["characteristic_label"] == config.characteristic_labels_ru[
            "domain_knowledge"
        ]
        assert row["confidence"] == pytest.approx(0.9)

    def test_llm_none_falls_back_to_nli_per_item(self, monkeypatch, tmp_path) -> None:
        config = _enable_llm(tmp_path, monkeypatch)

        resp = CharacteristicScoresResponse(characteristics=[
            CharacteristicScore(characteristic_id="experimental", confidence=0.9),
        ])

        async def fake_run_many(self, prompts, schema) -> list:
            assert len(prompts) == 2
            return [resp, None]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)
        monkeypatch.setattr(
            "krm.phase_2_5_characteristics._build_characteristic_classifier",
            lambda _config: _fake_nli_classifier,
        )

        df = extract_characteristics(
            config, ["lab work", "data analysis"], ["v1", "v2"]
        )

        assert set(df["vacancy_id"]) == {"v1", "v2"}

        v1 = df[df["vacancy_id"] == "v1"].iloc[0]
        assert v1["characteristic_id"] == "experimental"
        assert v1["confidence"] == pytest.approx(0.9)

        # v2's LLM call returned None → NLI fallback entailed the first hypothesis.
        v2 = df[df["vacancy_id"] == "v2"].iloc[0]
        assert v2["characteristic_id"] == config.characteristic_ids[0]
        assert v2["confidence"] == pytest.approx(0.9)
