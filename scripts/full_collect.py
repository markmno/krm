#!/usr/bin/env python3
"""Background collection script: Wayback hh.ru (2017-2026) + Trudvsem 4 regions.

Runs both collectors and merges into main DuckDB.
Logs to /tmp/krm_full_collect.log
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from krm.config import Config
from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback
from krm.phase0.storage import merge_into_main

cfg = Config()
start = time.time()

log = open("/tmp/krm_full_collect.log", "w", buffering=1)

def log_msg(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log.write(line + "\n")

# ── Phase 0: Wayback hh.ru 2017-2026 ──────────────────────────────
log_msg("=== Phase 0: Wayback hh.ru historical collection ===")
wayback_total = 0
for year in range(2017, 2027):
    try:
        log_msg(f"Starting hh.ru Wayback {year}...")
        stored = collect_hhru_wayback(year, cfg)
        wayback_total += stored
        log_msg(f"  {year}: {stored} stored (total: {wayback_total})")
    except Exception as e:
        log_msg(f"  {year}: FAILED — {e}")

elapsed = time.time() - start
log_msg(f"Wayback complete: {wayback_total} records in {elapsed/60:.0f} min")

# ── Merge into main DB ────────────────────────────────────────────
log_msg("=== Merging Phase 0 into main DB ===")
try:
    n = merge_into_main(cfg.phase0_db_path)
    log_msg(f"Merged {n} records into main DB")
except Exception as e:
    log_msg(f"Merge failed: {e}")

# ── Show final stats ──────────────────────────────────────────────
import duckdb
conn = duckdb.connect(str(cfg.output_dir / "krm.duckdb"), read_only=True)
try:
    total = conn.execute("SELECT COUNT(*) FROM raw_vacancies").fetchone()[0]
    log_msg(f"Main DB total: {total} raw vacancies")
    elapsed = time.time() - start
    log_msg(f"Total time: {elapsed/60:.0f} min")
finally:
    conn.close()

log.close()
