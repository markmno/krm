"""LLM-based skill extraction from Russian job-vacancy descriptions.

Talks to an OpenAI-compatible HTTP endpoint (llama-server / vLLM) with a
structured JSON prompt.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

_MAX_DESC_CHARS = 3000

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
    client: httpx.AsyncClient,
    desc: str,
    sem: asyncio.Semaphore,
    model: str,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    async with sem:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": desc[:_MAX_DESC_CHARS]},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_skills(text)


def extract_skills(
    descriptions: list[str],
    *,
    base_url: str,
    model: str,
    concurrency: int = 16,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> list[list[str]]:
    """Extract skill lists from a batch of Russian vacancy descriptions."""
    sem = asyncio.Semaphore(concurrency)

    async def _run() -> list[list[str]]:
        async with httpx.AsyncClient(base_url=base_url, timeout=180.0) as client:
            tasks = [
                _extract_one(client, d, sem, model, temperature, max_tokens)
                for d in descriptions
            ]
            return await asyncio.gather(*tasks)

    return asyncio.run(_run())
