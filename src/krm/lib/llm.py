"""Shared LLM client for the KRM pipeline.

Talks to an OpenAI-compatible HTTP endpoint (a local llama.cpp ``llama-server``)
using ``instructor`` + Pydantic structured output, with a 3-tier decode
fallback and an on-disk response cache for reproducibility.

``instructor`` and ``openai`` are imported lazily (inside the client methods),
so ``import krm.lib.llm`` stays cheap and the rest of the test suite does not
pay for a heavy dependency it never touches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from loguru import logger
from pydantic import BaseModel, TypeAdapter

if TYPE_CHECKING:
    from instructor import AsyncInstructor
    from openai import AsyncOpenAI
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
    )

    from krm.config import Config

T = TypeVar("T", bound=BaseModel)


class LLMPrompt(BaseModel):
    """A single LLM call: a system instruction plus a user payload."""

    system: str
    user: str


class LLMClient:
    """Async client for structured extraction over an OpenAI-compatible endpoint.

    Each call runs a 3-tier decode fallback:

    1. ``instructor`` JSON-schema structured output (primary);
    2. raw ``response_format={"type": "json_object"}`` + ``TypeAdapter`` validation;
    3. ``None`` — the caller maps ``None`` to its deterministic fallback.

    Responses are cached on disk (keyed by model/schema/temperature/prompt) so
    identical requests are reproducible across runs.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        concurrency: int = 16,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 2,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.base_url: str = base_url
        self.model: str = model
        self.concurrency: int = concurrency
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.max_retries: int = max_retries
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._raw_client: AsyncOpenAI | None = None
        self._instructor_client: AsyncInstructor | None = None

    # -- Lazy client construction ------------------------------------------------

    def _ensure_clients(self) -> tuple[AsyncOpenAI, AsyncInstructor]:
        """Build the raw + instructor clients once, on first use."""
        raw = self._raw_client
        instructor_client = self._instructor_client
        if raw is None or instructor_client is None:
            import instructor
            from openai import AsyncOpenAI

            # llama-server ignores the API key but the openai client requires one.
            raw = AsyncOpenAI(base_url=self.base_url, api_key="not-needed")
            instructor_client = instructor.from_openai(
                raw, mode=instructor.Mode.JSON_SCHEMA
            )
            self._raw_client = raw
            self._instructor_client = instructor_client
        return raw, instructor_client

    # -- Disk response cache -----------------------------------------------------

    def _cache_key(self, prompt: LLMPrompt, schema: type[T]) -> str:
        raw = (
            f"{self.model}|{schema.__name__}|{self.temperature}"
            f"|{prompt.system}|{prompt.user}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str, schema: type[T]) -> T | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301 — trusted local cache
        return schema.model_validate(data)

    def _cache_set(self, key: str, model: BaseModel) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model.model_dump(), f)

    # -- Decode (3-tier fallback) ------------------------------------------------

    async def _decode(self, prompt: LLMPrompt, schema: type[T]) -> T | None:
        from instructor.core import (
            IncompleteOutputException,
            InstructorRetryException,
        )
        from openai import APIError

        raw, instructor_client = self._ensure_clients()

        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": prompt.system,
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt.user,
        }
        messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]

        # Tier 1: instructor JSON-schema structured output (primary).
        try:
            result = await instructor_client.chat.completions.create(
                response_model=schema,
                messages=messages,
                max_retries=self.max_retries,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return result
        except (InstructorRetryException, IncompleteOutputException) as exc:
            logger.warning("LLM tier-1 (instructor JSON_SCHEMA) failed: {}", exc)

        # Tier 2: raw json_object + TypeAdapter validation.
        try:
            resp = await raw.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if content is None:
                logger.warning("LLM tier-2 (raw json_object) returned empty content")
                return None
            return TypeAdapter(schema).validate_python(json.loads(content))
        except (APIError, ValueError) as exc:
            logger.warning("LLM tier-2 (raw json_object) failed: {}", exc)

        # Tier 3: total failure — the caller maps None to its fallback.
        return None

    # -- Public API ---------------------------------------------------------------

    async def run_one(self, prompt: LLMPrompt, schema: type[T]) -> T | None:
        """Run a single structured extraction, returning ``None`` on failure."""
        key = self._cache_key(prompt, schema)
        cached = self._cache_get(key, schema)
        if cached is not None:
            return cached

        async with self._semaphore:
            result = await self._decode(prompt, schema)

        if result is not None:
            self._cache_set(key, result)
        return result

    async def run_many(self, prompts: list[LLMPrompt], schema: type[T]) -> list[T | None]:
        """Run a batch of extractions, preserving input order.

        Concurrency is bounded by the client's semaphore; ``asyncio.gather``
        returns results in input order.
        """
        return await asyncio.gather(*(self.run_one(p, schema) for p in prompts))


def build_llm(config: Config) -> LLMClient:
    """Build an :class:`LLMClient` from a ``Config`` (or config-like) object."""
    return LLMClient(
        base_url=config.llm_base_url,
        model=config.llm_model,
        concurrency=config.llm_concurrency,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        max_retries=config.llm_max_retries,
        cache_dir=config.llm_cache_dir,
    )
