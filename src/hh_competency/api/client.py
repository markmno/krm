"""Async HTTP client for HH.ru API.

API docs: https://github.com/hhru/api

Key endpoints:
- GET /vacancies — search with text, area, professional_role, per_page, page
- GET /vacancies/{id} — full vacancy details
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from hh_competency.api.rate_limiter import RateLimiter, RetryHandler
from hh_competency.storage.models import VacancyData


class HHClient:
    """Async HTTP client for the HH.ru API.

    Handles rate limiting, retries, pagination, and response parsing.
    """

    def __init__(
        self,
        base_url: str = "https://api.hh.ru",
        user_agent: str = "hh-competency/0.1.0",
        requests_per_second: float = 2.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"User-Agent": user_agent}
        self._limiter = RateLimiter(requests_per_second)
        self._retry = RetryHandler(max_retries=max_retries, backoff=retry_backoff)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> HHClient:
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(30.0),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HHClient must be used as async context manager")
        return self._client

    async def search_vacancies(
        self,
        text: str,
        area: int = 113,
        professional_role: int | None = None,
        per_page: int = 100,
        page: int = 0,
    ) -> dict[str, Any]:
        """Search for vacancies on HH.ru.

        Args:
            text: Search query text (fuzzy matched by HH.ru).
            area: HH.ru area ID. 113 = Россия, 1 = Москва.
            professional_role: HH.ru professional role ID (optional filter).
            per_page: Results per page (max 100).
            page: Page number (0-indexed).

        Returns:
            Raw API response dict with 'items', 'pages', 'found' keys.
        """
        params: dict[str, Any] = {
            "text": text,
            "area": area,
            "per_page": per_page,
            "page": page,
        }
        if professional_role is not None:
            params["professional_role"] = professional_role

        await self._limiter.acquire()

        async def _do() -> dict[str, Any]:
            assert self._client is not None
            response = await self._client.get(
                f"{self._base_url}/vacancies", params=params
            )
            response.raise_for_status()
            return response.json()

        return await self._retry.execute(_do)

    async def get_vacancy_detail(self, vacancy_id: str) -> dict[str, Any]:
        """Fetch full vacancy details including description and key_skills.

        Args:
            vacancy_id: HH.ru vacancy ID.

        Returns:
            Full vacancy API response.
        """
        await self._limiter.acquire()

        async def _do() -> dict[str, Any]:
            assert self._client is not None
            response = await self._client.get(
                f"{self._base_url}/vacancies/{vacancy_id}"
            )
            response.raise_for_status()
            return response.json()

        return await self._retry.execute(_do)

    async def scrape_specialty(
        self,
        search_keywords: list[str],
        professional_roles: list[int] | None = None,
        area: int = 113,
        max_pages: int = 20,
    ) -> list[VacancyData]:
        """Full scrape workflow: search → paginate → deduplicate → get details.

        Args:
            search_keywords: List of search query strings.
            professional_roles: HH.ru professional role IDs.
            area: HH.ru area ID.
            max_pages: Maximum pages to fetch per keyword (100 per page).

        Returns:
            List of complete VacancyData objects, deduplicated by ID.
        """
        # Phase 1: Search all keywords, collect vacancy IDs
        seen_ids: set[str] = set()
        vacancy_briefs: list[dict[str, Any]] = []

        for keyword in search_keywords:
            for page in range(max_pages):
                search_result = await self.search_vacancies(
                    text=keyword,
                    area=area,
                    per_page=100,
                    page=page,
                )

                items = search_result.get("items", [])
                if not items:
                    break  # No more results for this keyword

                for item in items:
                    vac_id = str(item["id"])
                    if vac_id not in seen_ids:
                        seen_ids.add(vac_id)
                        vacancy_briefs.append(item)

                if page + 1 >= search_result.get("pages", 0):
                    break  # No more pages for this keyword

        # Phase 2: Fetch full details in parallel (semaphore=3)
        semaphore = asyncio.Semaphore(3)

        async def _fetch_detail(brief: dict[str, Any]) -> VacancyData:
            async with semaphore:
                detail = await self.get_vacancy_detail(brief["id"])
                return VacancyData.from_api_response(detail)

        vacancies = await asyncio.gather(
            *[_fetch_detail(b) for b in vacancy_briefs]
        )

        return list(vacancies)
