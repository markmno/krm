"""LLM-based skill extraction from Russian job-vacancy descriptions.

Calls a locally-served vLLM OpenAI-compatible endpoint (Qwen2.5-7B-Instruct)
with a structured JSON prompt. vLLM is used instead of loading the model
directly via transformers, which crashes on the 14 GB checkpoint under the
bleeding-edge transformers/torch stack.

Usage:
    from krm.lib.llm_skills import extract_skills
    skill_lists = extract_skills(["описание вакансии 1", "описание 2", ...])
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

_BASE_URL = "http://localhost:8000/v1"
_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_CONCURRENCY = 16

_SYSTEM_PROMPT = (
    "Ты — эксперт по анализу вакансий. Извлеки из описания вакансии все "
    "профессиональные навыки и компетенции (hard skills), которые требуются от "
    "кандидата. Не включай условия труда, зарплату, бенефиты, график работы, "
    "информацию о компании, требования к образованию или стажу. Каждый навык — "
    "короткая фраза из 1–4 слов на русском. Верни строго JSON без пояснений: "
    '{"skills": ["навык1", "навык2", ...]}'
)


def _parse_skills(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    skills = obj.get("skills", [])
    return [s.strip() for s in skills if isinstance(s, str) and s.strip()]


async def _extract_one(
    client: httpx.AsyncClient, desc: str, sem: asyncio.Semaphore
) -> list[str]:
    async with sem:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": desc[:3000]},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_skills(text)


def extract_skills(descriptions: list[str]) -> list[list[str]]:
    """Extract skill lists from a batch of Russian vacancy descriptions."""
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _run() -> list[list[str]]:
        async with httpx.AsyncClient(base_url=_BASE_URL, timeout=180.0) as client:
            tasks = [_extract_one(client, d, sem) for d in descriptions]
            return await asyncio.gather(*tasks)

    return asyncio.run(_run())
