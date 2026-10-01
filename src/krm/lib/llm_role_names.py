"""LLM-based human-readable naming for discovered role clusters.

Post-processes the deterministic ``roles.parquet`` output of Phase 3 by asking
the shared :class:`~krm.lib.llm.LLMClient` for a concise, human-readable role
name per non-noise cluster. Clustering itself is never re-run here.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
from pydantic import BaseModel

from krm.config import Config
from krm.lib.llm import LLMPrompt, build_llm

_ROLE_NAME_SYSTEM_PROMPT = (
    "Ты — эксперт по рынку труда и классификатору профессий. Дай ёмкое, "
    "человекочитаемое название профессиональной роли (на русском языке) на "
    "основе центральных должностей кластера и его характеристического профиля. "
    "Верни только короткое имя роли (2–6 слов), без пояснений."
)


class RoleNameResponse(BaseModel):
    """Structured LLM output: a concise human-readable role name."""

    label: str


def _build_role_name_prompt(top_titles: list[str], profile: dict[str, float]) -> str:
    """Assemble the LLM user payload for naming a single role."""
    titles = ", ".join(top_titles)
    profile_json = json.dumps(profile, ensure_ascii=False)
    return (
        f"Центральные должности кластера: {titles}.\n"
        f"Характеристический профиль роли: {profile_json}.\n"
        "Дай одно короткое название этой роли на русском."
    )


def name_roles(roles_df: pd.DataFrame, config: Config) -> pd.Series:
    """Name each non-noise role via the LLM, falling back to ``role_label``.

    Builds one :class:`LLMPrompt` per non-noise role from its top-3 central
    titles and characteristic profile, then runs the batch through the shared
    ``LLMClient``. A ``None`` result (LLM failure) keeps the deterministic
    ``role_label``; the noise row always keeps its ``role_label``.
    """
    labels: list[str] = []
    prompts: list[LLMPrompt] = []
    prompt_positions: list[int] = []

    for pos in range(len(roles_df)):
        row = roles_df.iloc[pos]
        default_label = str(row["role_label"])
        labels.append(default_label)
        if bool(row["noise_flag"]):
            continue
        top_titles: list[str] = json.loads(row["top_titles"])
        profile: dict[str, float] = json.loads(row["characteristic_profile"])
        prompts.append(
            LLMPrompt(
                system=_ROLE_NAME_SYSTEM_PROMPT,
                user=_build_role_name_prompt(top_titles, profile),
            )
        )
        prompt_positions.append(pos)

    if prompts:
        client = build_llm(config)
        results = asyncio.run(client.run_many(prompts, RoleNameResponse))
        for pos, result in zip(prompt_positions, results, strict=True):
            if result is not None and result.label.strip():
                labels[pos] = result.label.strip()

    return pd.Series(labels, index=roles_df.index)
