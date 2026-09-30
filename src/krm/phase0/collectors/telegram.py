"""Telegram channel collector — orchestrates fetch → parse → map → store.

Reads vacancy-style messages from configured Telegram channels using telethon,
then reuses the Phase 0 schema pipeline (parse → map → normalize → enrich →
upsert) shared by all sources.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from krm.phase0.parsers.telegram import parse_telegram_message
from krm.phase0.schema import enrich_record, map_telegram, normalize_record
from krm.phase0.storage import (
    finish_phase0_run,
    get_phase0_connection,
    init_phase0_tables,
    start_phase0_run,
    upsert_phase0_vacancy,
)

if TYPE_CHECKING:
    from krm.config import Config

logger = logging.getLogger(__name__)


def _message_to_record(
    text: str,
    channel: str,
    message_id: int,
    date: str | None,
    capture_ts: str,
) -> dict[str, Any]:
    """Parse → map → normalize → enrich a single Telegram message.

    Pure (no network) so it can be unit-tested without a live session.
    """
    parsed = parse_telegram_message(
        text, channel=channel, message_id=message_id, date=date
    )
    mapped = map_telegram(parsed, capture_ts)
    normalized = normalize_record(mapped)
    return enrich_record(normalized)


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    """Return True if no keywords configured, or any keyword (case-insensitive) appears."""
    if not keywords:
        return True
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def collect_telegram(config: Config | None = None) -> int:
    """Collect vacancy messages from configured Telegram channels.

    Reads the ``telegram`` section of the config; if ``enabled`` is false,
    logs and returns 0 without touching the network. Otherwise creates a
    ``telethon.TelegramClient`` and drives the async collection with the
    client's own event loop.

    Note: ``client.start()`` is a synchronous telethon entry point that
    connects and authenticates (on first run it performs interactive
    phone-number + code authorization via stdin; subsequent runs reuse the
    persisted ``.session`` file). The async collection is then run with
    ``client.loop.run_until_complete``.

    Args:
        config: Pipeline configuration.

    Returns:
        Number of vacancies successfully stored.
    """
    if config is None:
        from krm.config import Config

        config = Config()

    if not config.phase0_telegram_enabled:
        logger.info("Telegram collection disabled; skipping")
        return 0

    session_name = config.phase0_telegram_session_name
    api_id = config.phase0_telegram_api_id
    api_hash = config.phase0_telegram_api_hash
    channels = config.phase0_telegram_channels
    keywords = config.phase0_telegram_keywords
    limit = config.phase0_telegram_limit_per_channel

    run_id = f"telegram-{datetime.now().strftime('%m%d_%H%M')}"

    conn = get_phase0_connection()
    init_phase0_tables(conn)
    start_phase0_run(conn, run_id, "telegram", ",".join(channels))

    fetched = 0
    stored = 0

    try:
        from telethon import TelegramClient

        client = TelegramClient(session_name, api_id, api_hash)
        client.start()

        async def _collect() -> tuple[int, int]:
            local_fetched = 0
            local_stored = 0
            for channel in channels:
                try:
                    entity = await client.get_entity(channel)
                except Exception:
                    logger.warning("Failed to resolve Telegram channel %r", channel)
                    continue

                try:
                    messages_raw = await client.get_messages(entity, limit=limit)
                except Exception:
                    logger.warning("Failed to fetch messages from %r", channel)
                    continue

                if messages_raw is None:
                    continue
                messages = (
                    [messages_raw] if not isinstance(messages_raw, list) else messages_raw
                )

                for message in messages:
                    text = getattr(message, "message", None)
                    if not text:
                        continue
                    if not _matches_keywords(text, keywords):
                        continue

                    local_fetched += 1
                    message_date = getattr(message, "date", None)
                    date = message_date.isoformat() if message_date else None
                    capture_ts = datetime.now().isoformat(timespec="seconds")
                    try:
                        record = _message_to_record(
                            text, channel, message.id, date, capture_ts
                        )
                        upsert_phase0_vacancy(conn, run_id, record["id"], record)
                        local_stored += 1
                    except Exception:
                        logger.warning(
                            "Skipping Telegram message %s in %r",
                            getattr(message, "id", "?"),
                            channel,
                        )
            return local_fetched, local_stored

        try:
            fetched, stored = client.loop.run_until_complete(_collect())
        finally:
            client.disconnect()
    finally:
        finish_phase0_run(conn, run_id, fetched, stored)
        logger.info("Completed %s: fetched=%d stored=%d", run_id, fetched, stored)
        conn.close()

    return stored
