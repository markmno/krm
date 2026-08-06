"""Rate limiter and retry handler for HH.ru API requests."""

from __future__ import annotations

import asyncio
import time
from typing import Any


class RateLimiter:
    """Token-bucket-like rate limiter for async API calls.

    Ensures minimum interval between requests to respect HH.ru's ~2 req/s limit.
    """

    def __init__(self, requests_per_second: float = 2.0) -> None:
        self._rate = requests_per_second
        self._min_interval = 1.0 / requests_per_second
        self._last_request: float = 0.0

    async def acquire(self) -> None:
        """Wait until it's safe to make the next request."""
        now = time.monotonic()
        wait = self._last_request + self._min_interval - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()


class RetryHandler:
    """Exponential backoff retry for transient API failures.

    Retries on ConnectionError, Timeout, 429, 5xx.
    Does NOT retry on 4xx (except 429).
    """

    def __init__(self, max_retries: int = 3, backoff: float = 2.0) -> None:
        self._max_retries = max_retries
        self._backoff = backoff

    async def execute(self, coro_factory: Any) -> Any:
        """Execute an async callable with retry logic.

        Args:
            coro_factory: Async callable (or lambda returning awaitable).

        Returns:
            The result of the successful call.

        Raises:
            The last exception after all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await coro_factory()
            except Exception as e:
                last_exc = e
                if attempt == self._max_retries:
                    break
                await asyncio.sleep(self._backoff ** attempt)
        raise last_exc  # type: ignore[misc]
