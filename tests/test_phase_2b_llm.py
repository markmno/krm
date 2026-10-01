"""Tests for the phase-2b skill-extraction dispatch (LLM vs spaCy fallback).

Verifies that ``extract_skills``' extraction step dispatches on
``config.use_llm_phase_2b``:

* ``use_llm: true`` → the shared ``LLMClient`` (via ``run_many``) drives
  structured extraction, and its output is mapped to ``(text, lemma)`` tuples.
* a ``None`` item from ``run_many`` (total failure for that vacancy) maps to an
  empty skills list.
* ``use_llm: false`` → the deterministic spaCy noun-phrase path is used and the
  LLM client is never touched.

No network, spaCy, or DuckDB is used: ``LLMClient.run_many`` and the noun-phrase
extractor are both monkeypatched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from krm.phase_2b_extract_skills import (
    SkillsResponse,
    _extract_skills_llm,
    _extract_skills_per_desc,
)


def _llm_config(use_llm: bool) -> MagicMock:
    config = MagicMock()
    type(config).use_llm_phase_2b = PropertyMock(return_value=use_llm)
    type(config).llm_base_url = PropertyMock(return_value="http://localhost:8080/v1")
    type(config).llm_model = PropertyMock(return_value="Qwen/Qwen3-8B-Instruct")
    type(config).llm_concurrency = PropertyMock(return_value=4)
    type(config).llm_temperature = PropertyMock(return_value=0.0)
    type(config).llm_max_tokens = PropertyMock(return_value=256)
    type(config).llm_max_retries = PropertyMock(return_value=2)
    type(config).llm_cache_dir = PropertyMock(return_value=None)
    return config


class TestSkillsResponse:
    def test_normalizes_skills(self) -> None:
        resp = SkillsResponse(
            skills=[
                " Python ",
                "",
                "машинное обучение",
                "МАШИННОЕ ОБУЧЕНИЕ",
                "a b c d e",  # 5 words → dropped
                123,  # non-str → dropped
                "один",
            ]
        )
        assert resp.skills == ["python", "машинное обучение", "один"]

    def test_non_list_input_yields_empty(self) -> None:
        assert SkillsResponse.model_validate({"skills": "python"}).skills == []

    def test_default_is_empty(self) -> None:
        assert SkillsResponse().skills == []


class TestDispatchUseLlm:
    def test_use_llm_true_uses_llm_and_maps_output(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        async def fake_run_many(self, prompts, schema):
            captured["schema"] = schema
            captured["prompts"] = prompts
            return [
                SkillsResponse(skills=[" Python ", "Машинное обучение"]),
                None,
            ]

        monkeypatch.setattr("krm.lib.llm.LLMClient.run_many", fake_run_many)

        result = _extract_skills_per_desc(
            ["описание раз", "описание два"], _llm_config(use_llm=True)
        )

        assert captured["schema"] is SkillsResponse
        assert len(captured["prompts"]) == 2  # type: ignore[arg-type]
        assert result == [
            [("python", "python"), ("машинное обучение", "машинное обучение")],
            [],
        ]

    def test_none_item_maps_to_empty_skills(self, monkeypatch) -> None:
        async def fake_run_many(self, prompts, schema):
            return [None]

        monkeypatch.setattr("krm.lib.llm.LLMClient.run_many", fake_run_many)

        result = _extract_skills_llm(["описание"], _llm_config(use_llm=True))

        assert result == [[]]


class TestDispatchUseNounPhrases:
    def test_use_llm_false_uses_noun_phrases(self, monkeypatch) -> None:
        import krm.phase_2b_extract_skills as mod

        def fake_noun_phrases(descriptions):
            return [[("анализ данных", "анализ данные")], []]

        def explode(descriptions, config):
            raise AssertionError("LLM path must not be used when use_llm is false")

        monkeypatch.setattr(mod, "_extract_skills_noun_phrases", fake_noun_phrases)
        monkeypatch.setattr(mod, "_extract_skills_llm", explode)

        result = mod._extract_skills_per_desc(["a", "b"], _llm_config(use_llm=False))

        assert result == [[("анализ данных", "анализ данные")], []]
