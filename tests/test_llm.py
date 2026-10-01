"""Unit tests for the shared LLM client (no network).

Mocks ``instructor.from_openai`` and the wrapped ``openai.AsyncOpenAI`` client
to verify the 3-tier decode fallback, batch ordering/concurrency, the disk
response cache, and ``build_llm`` config wiring.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from krm.lib.llm import LLMClient, LLMPrompt, build_llm


class Skills(BaseModel):
    skills: list[str]


BASE_URL = "http://localhost:8080/v1"
MODEL = "Qwen/Qwen3.8-27B"


def _retry_exc() -> Exception:
    from instructor.core import InstructorRetryException

    return InstructorRetryException("boom", n_attempts=1, total_usage=None)


def _json_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_tier1_returns_model() -> None:
    """Tier 1 (instructor JSON_SCHEMA) returns the Pydantic model on success."""
    expected = Skills(skills=["python", "sql"])

    instructor_client = MagicMock()
    instructor_client.chat.completions.create = AsyncMock(return_value=expected)
    raw = MagicMock()

    with patch("openai.AsyncOpenAI", return_value=raw), patch(
        "instructor.from_openai", return_value=instructor_client
    ):
        client = LLMClient(BASE_URL, MODEL)
        result = asyncio.run(client.run_one(LLMPrompt(system="s", user="u"), Skills))

    assert result == expected


def test_tier2_json_object_fallback() -> None:
    """Tier 2 (raw json_object) fires when tier 1 raises InstructorRetryException."""
    instructor_client = MagicMock()
    instructor_client.chat.completions.create = AsyncMock(side_effect=_retry_exc())

    raw = MagicMock()
    raw.chat.completions.create = AsyncMock(
        return_value=_json_response(json.dumps({"skills": ["java", "c++"]}))
    )

    with patch("openai.AsyncOpenAI", return_value=raw), patch(
        "instructor.from_openai", return_value=instructor_client
    ):
        client = LLMClient(BASE_URL, MODEL)
        result = asyncio.run(client.run_one(LLMPrompt(system="s", user="u"), Skills))

    assert result == Skills(skills=["java", "c++"])
    raw.chat.completions.create.assert_awaited_once()


def test_tier3_returns_none_on_total_failure() -> None:
    """Tier 3 returns None when both structured paths fail (invalid JSON)."""
    instructor_client = MagicMock()
    instructor_client.chat.completions.create = AsyncMock(side_effect=_retry_exc())

    raw = MagicMock()
    raw.chat.completions.create = AsyncMock(
        return_value=_json_response("this is not json")
    )

    with patch("openai.AsyncOpenAI", return_value=raw), patch(
        "instructor.from_openai", return_value=instructor_client
    ):
        client = LLMClient(BASE_URL, MODEL)
        result = asyncio.run(client.run_one(LLMPrompt(system="s", user="u"), Skills))

    assert result is None


def test_run_many_order_and_concurrency() -> None:
    """run_many preserves input order and bounds in-flight requests."""
    prompts = [LLMPrompt(system="s", user=f"u{i}") for i in range(6)]

    state = {"active": 0, "max": 0}

    async def fake_create(response_model=None, messages=None, **kwargs):
        user = messages[1]["content"]
        state["active"] += 1
        state["max"] = max(state["max"], state["active"])
        await asyncio.sleep(0.005)
        state["active"] -= 1
        return response_model(skills=[user])

    instructor_client = MagicMock()
    instructor_client.chat.completions.create = AsyncMock(side_effect=fake_create)
    raw = MagicMock()

    with patch("openai.AsyncOpenAI", return_value=raw), patch(
        "instructor.from_openai", return_value=instructor_client
    ):
        client = LLMClient(BASE_URL, MODEL, concurrency=2)
        results = asyncio.run(client.run_many(prompts, Skills))

    assert [r.skills for r in results] == [[f"u{i}"] for i in range(6)]
    assert state["max"] == 2


def test_cache_hit_skips_network(tmp_path: Path) -> None:
    """A cache hit returns the cached model without touching the network."""
    cache_dir = tmp_path / "llm_cache"
    expected = Skills(skills=["cached"])

    instructor_client = MagicMock()
    instructor_client.chat.completions.create = AsyncMock(return_value=expected)
    raw = MagicMock()

    prompt = LLMPrompt(system="s", user="u")

    with patch("openai.AsyncOpenAI", return_value=raw), patch(
        "instructor.from_openai", return_value=instructor_client
    ):
        client = LLMClient(BASE_URL, MODEL, cache_dir=cache_dir)
        first = asyncio.run(client.run_one(prompt, Skills))
    assert first == expected

    # Fresh client, same cache dir: the network must not be touched.
    second_instructor = MagicMock()
    second_instructor.chat.completions.create = AsyncMock(
        return_value=Skills(skills=["SHOULD NOT BE USED"])
    )
    with patch("openai.AsyncOpenAI") as mock_aclient, patch(
        "instructor.from_openai", return_value=second_instructor
    ) as mock_from_openai:
        client2 = LLMClient(BASE_URL, MODEL, cache_dir=cache_dir)
        second = asyncio.run(client2.run_one(prompt, Skills))

    assert second == expected
    mock_aclient.assert_not_called()
    mock_from_openai.assert_not_called()
    second_instructor.chat.completions.create.assert_not_awaited()


def test_build_llm_wires_config(tmp_path: Path) -> None:
    """build_llm maps config properties onto the LLMClient constructor."""
    config = MagicMock()
    config.llm_base_url = "http://example:8080/v1"
    config.llm_model = "my-model"
    config.llm_concurrency = 3
    config.llm_temperature = 0.5
    config.llm_max_tokens = 128
    config.llm_max_retries = 4
    config.llm_cache_dir = str(tmp_path / "cache")

    client = build_llm(config)

    assert client.base_url == "http://example:8080/v1"
    assert client.model == "my-model"
    assert client.concurrency == 3
    assert client.temperature == 0.5
    assert client.max_tokens == 128
    assert client.max_retries == 4
    assert client._cache_dir == tmp_path / "cache"


def test_config_properties_default_when_absent(tmp_path: Path) -> None:
    """Config LLM properties return safe defaults when the llm section is absent."""
    from krm.config import Config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("pipeline:\n  raw_dir: data/raw\n")

    config = Config(path=cfg_file)

    assert config.llm_enabled is True
    assert config.llm_max_retries == 2
    assert config.llm_response_format == "json_schema"
    assert config.llm_cache_dir == "data/llm_cache"
    assert config.use_llm_phase_2b is False
    assert config.use_llm_phase_2_5 is False
    assert config.use_llm_phase_5 is False
    assert config.use_llm_phase_5b is False
    assert config.use_llm_phase_3 is False
