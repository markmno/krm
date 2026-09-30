#!/usr/bin/env python3
"""Trudvsem region-based bulk collector — 60 Russian regions, no text search.

Stores directly in the main pipeline DuckDB (data/krm.duckdb).
Uses region_code partitioning to bypass the 200-item offset cap.

Usage:
    PYTHONPATH=src python3 scripts/collect_regions.py            # full collection
    PYTHONPATH=src python3 scripts/collect_regions.py --dry-run  # census only
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from krm.lib.io import get_connection, init_tables, upsert_raw_vacancy
from krm.phase0.collectors.trudvsem import TrudvsemCollector

REGIONS = {
    "7700000000000": "Москва",
    "5000000000000": "Московская область",
    "7800000000000": "Санкт-Петербург",
    "4700000000000": "Ленинградская область",
    "6600000000000": "Свердловская область",
    "5400000000000": "Новосибирская область",
    "1600000000000": "Татарстан",
    "2300000000000": "Краснодарский край",
    "7400000000000": "Челябинская область",
    "5200000000000": "Нижегородская область",
    "6300000000000": "Самарская область",
    "0200000000000": "Башкортостан",
    "6100000000000": "Ростовская область",
    "5900000000000": "Пермский край",
    "5500000000000": "Омская область",
    "3800000000000": "Иркутская область",
    "2400000000000": "Красноярский край",
    "8600000000000": "Ханты-Мансийский АО",
    "3600000000000": "Воронежская область",
    "3400000000000": "Волгоградская область",
    "4200000000000": "Кемеровская область",
    "5600000000000": "Оренбургская область",
    "2200000000000": "Алтайский край",
    "6400000000000": "Саратовская область",
    "7300000000000": "Ульяновская область",
    "1800000000000": "Удмуртия",
    "2700000000000": "Хабаровский край",
    "2500000000000": "Приморский край",
    "7200000000000": "Тюменская область",
    "3700000000000": "Ивановская область",
    "6200000000000": "Рязанская область",
    "3300000000000": "Владимирская область",
    "4000000000000": "Калужская область",
    "4600000000000": "Курская область",
    "3200000000000": "Брянская область",
    "7100000000000": "Тульская область",
    "7600000000000": "Ярославская область",
    "2900000000000": "Архангельская область",
    "3500000000000": "Вологодская область",
    "3900000000000": "Калининградская область",
    "1000000000000": "Карелия",
    "1100000000000": "Коми",
    "2600000000000": "Ставропольский край",
    "3000000000000": "Астраханская область",
    "6800000000000": "Тамбовская область",
    "5800000000000": "Пензенская область",
    "2100000000000": "Чувашия",
    "1200000000000": "Марий Эл",
    "1300000000000": "Мордовия",
    "7500000000000": "Забайкальский край",
    "2800000000000": "Амурская область",
    "6500000000000": "Сахалинская область",
    "4100000000000": "Камчатский край",
    "4900000000000": "Магаданская область",
    "1400000000000": "Якутия",
    "8700000000000": "Чукотский АО",
    "8300000000000": "Ненецкий АО",
    "8900000000000": "Ямало-Ненецкий АО",
}

DRY_RUN = "--dry-run" in sys.argv

def main():
    conn = get_connection()
    init_tables(conn)
    run_id = f"regions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    c = TrudvsemCollector(rate_limit_rps=0.3)
    total = 0
    start = time.time()
    seen_ids: set[str] = set()

    try:
        for rc, rname in REGIONS.items():
            api_total = c.count(region_code=rc)
            print(f"\n[{rname}] API total: {api_total}")

            if api_total == 0:
                continue

            if DRY_RUN:
                total += min(api_total, 198)
                continue

            vacancies = c.collect_all(region_code=rc)
            stored = 0
            for v in vacancies:
                vid = v["id"]
                if vid not in seen_ids:
                    upsert_raw_vacancy(conn, run_id, vid, v)
                    seen_ids.add(vid)
                    stored += 1

            total += stored
            elapsed = time.time() - start
            print(f"  → stored {stored} (total: {total}, {total/elapsed*60:.0f}/min)")

    finally:
        c.close()

    elapsed = time.time() - start
    final = conn.execute("SELECT COUNT(*) FROM raw_vacancies").fetchone()[0]
    conn.close()

    print(f"\n{'=' * 60}")
    if DRY_RUN:
        print(f"DRY RUN — estimated: {total} records")
    else:
        print(f"DONE: {total} new + existing = {final} total in {elapsed/60:.0f} min")


if __name__ == "__main__":
    main()
