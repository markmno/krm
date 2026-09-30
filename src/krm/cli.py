"""CLI entry point for KRM pipeline.

Usage:
    krm collect         # Phase 1: Scrape HH.ru vacancies
    krm classify        # Phase 2: Classify STEM/IT/non-STEM
    krm roles           # Phase 3: Discover roles via clustering
    krm skills          # Phase 4: Extract skills per role
    krm extract-skills  # Phase 2b: Extract Russian skill phrases
    krm characteristics  # Phase 5: Map skills to competency characteristics
    krm soft             # Phase 5b: Classify role archetypes + soft competences
    krm model           # Phase 6: Build competency models + charts
    krm validate        # Phase 7: Validate and report
    krm report          # Phase 8: Generate the HTML report
    krm all             # Run all phases sequentially
    krm phase0-census   # Phase 0: Wayback CDX coverage census
    krm phase0-hhru     # Phase 0: Collect hh.ru Wayback data
    krm phase0-linkedin # Phase 0: Collect LinkedIn Wayback data
    krm phase0-trudvsem # Phase 0: Collect Trudvsem open data
    krm phase0-rostud   # Phase 0: Collect Rostrud bulk data
    krm phase0-all      # Phase 0: Run all collectors
    krm phase0-verify   # Phase 0: Verify collection run summaries
    krm phase0-merge    # Phase 0: Merge phase0 DB into main DB
"""

from __future__ import annotations

import sys
from pathlib import Path

from krm.config import Config

_DOMAIN: str | None = None


def _get_config(config_path: str | None = None) -> Config:
    path = Path(config_path) if config_path else None
    cfg = Config(path, domain=_DOMAIN)
    cfg.ensure_dirs()
    return cfg


def cmd_collect(config_path: str | None = None) -> None:
    """Phase 1: Scrape HH.ru vacancies from the public website (no API key)."""
    from krm.phase_1_site import collect_site

    cfg = _get_config(config_path)
    n = collect_site(cfg)
    print(f"Collected {n} new vacancies")


def cmd_collect_domain(config_path: str | None = None) -> None:
    """Phase 1: Collect vacancies for one domain (physics/biology/chemistry).

    Usage: krm collect-domain physics  →  data/physics.duckdb
    """
    from krm.phase_1_site import collect_site

    cfg = _get_config(config_path)
    domain = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    if domain not in cfg.domains:
        available = ", ".join(cfg.domains)
        print(f"Usage: krm collect-domain <domain>  (available: {available})")
        sys.exit(1)
    n = collect_site(cfg, domain=domain)
    print(f"Collected {n} new vacancies for domain '{domain}'")


def cmd_classify(config_path: str | None = None) -> None:
    """Phase 2: Classify STEM/IT/non-STEM."""
    from krm.phase_2_classify import classify_vacancies

    cfg = _get_config(config_path)
    df = classify_vacancies(cfg)
    counts = df["stem_category"].value_counts().to_dict()
    print("Classification results:")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")


def cmd_phase25(config_path: str | None = None) -> None:
    """Phase 2.5: Extract vacancy-level characteristics for downstream grounding."""
    from krm.phase_2_5_characteristics import run_pipeline

    cfg = _get_config(config_path)
    df, title_chars = run_pipeline(cfg)
    n_vacancies = df["vacancy_id"].nunique()
    n_characteristics = df["characteristic_id"].nunique()
    n_titles = len(title_chars)
    print(
        f"Extracted {len(df)} characteristic scores "
        f"for {n_vacancies} vacancies "
        f"({n_characteristics} characteristics, {n_titles} unique titles)"
    )


def cmd_roles(config_path: str | None = None) -> None:
    """Phase 3: Discover roles via HDBSCAN clustering."""
    from krm.phase_3_roles import discover_roles

    cfg = _get_config(config_path)
    df = discover_roles(cfg)
    n_roles = df[df["role_id"] != -1]["role_id"].nunique() if not df.empty else 0
    n_noise = (df["role_id"] == -1).sum() if not df.empty else 0
    print(f"Discovered {n_roles} roles ({n_noise} noise points)")
    if n_roles > 0:
        for _, row in df[df["role_id"] != -1].iterrows():
            print(f"  {row['role_id']}: {row['role_label']} ({row['member_count']} vacancies)")


def cmd_skills(config_path: str | None = None) -> None:
    """Phase 4: Extract skills per role."""
    from krm.phase_4_skills import extract_skills

    cfg = _get_config(config_path)
    df = extract_skills(cfg)
    n_skills = df["skill_canonical_name"].nunique()
    n_roles = df["role_id"].nunique()
    print(f"Extracted {n_skills} unique skills across {n_roles} roles")


def cmd_extract_skills(config_path: str | None = None) -> None:
    """Phase 2b: Extract Russian skill phrases from vacancy descriptions."""
    from krm.phase_2b_extract_skills import extract_skills

    cfg = _get_config(config_path)
    skills_df, vocab_df = extract_skills(cfg)
    print(f"Extracted {len(vocab_df)} unique skill phrases from {skills_df['vacancy_id'].nunique()} vacancies")
    print(f"Total skill-vacancy pairs: {len(skills_df)}")


def cmd_characteristics(config_path: str | None = None) -> None:
    """Phase 5: Map skills to competency characteristics."""
    from krm.phase_5_axes import map_to_characteristics

    cfg = _get_config(config_path)
    df = map_to_characteristics(cfg)
    if df.empty:
        print("No characteristic scores produced (no skill data available)")
        return
    print(f"Characteristic scores computed for {df['role_id'].nunique()} roles")
    for characteristic_id in cfg.characteristic_ids:
        label = cfg.characteristic_labels_ru[characteristic_id]
        mean_score = df[df["characteristic_id"] == characteristic_id]["proficiency"].mean()
        print(f"  {label}: mean={mean_score:.2f}")


def cmd_soft(config_path: str | None = None) -> None:
    """Phase 5b: Classify role archetypes and score soft competences."""
    from krm.phase_5b_soft import classify_role_archetypes, score_soft_competences

    cfg = _get_config(config_path)
    archetypes = classify_role_archetypes(cfg)
    scores = score_soft_competences(cfg)
    if archetypes.empty:
        print("No role archetypes produced (no skill data available)")
        return
    archetype_map = dict(zip(archetypes["role_id"], archetypes["archetype"], strict=True))
    soft_labels = cfg.soft_competence_labels_ru
    print(f"Soft competences computed for {scores['role_id'].nunique()} roles")
    for role_id in sorted(archetype_map):
        role_scores = scores[scores["role_id"] == role_id]
        parts: list[str] = []
        for soft_id in cfg.soft_competence_ids:
            label = soft_labels[soft_id]
            row = role_scores[role_scores["soft_id"] == soft_id]
            value = row["proficiency"].iloc[0] if not row.empty else float("nan")
            parts.append(f"{label}={value:.2f}")
        print(f"  role {role_id} [{archetype_map[role_id]}]: " + ", ".join(parts))


def cmd_model(config_path: str | None = None) -> None:
    """Phase 6: Build competency models and spider charts."""
    from krm.phase_6_model import build_models

    cfg = _get_config(config_path)
    models = build_models(cfg)
    if not models:
        print("No models built (no role data available)")
        return
    print(f"Built {len(models)} competency models")
    for m in models:
        print(f"  {m['role_id']}: {m['role_label']}")


def cmd_validate(config_path: str | None = None) -> None:
    """Phase 7: Validate pipeline and generate reports."""
    from krm.phase_7_validate import validate

    cfg = _get_config(config_path)
    report = validate(cfg)
    integration = report.get("phases", {}).get("integration", {})
    for k, v in integration.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
    for phase_name, phase_data in report.get("phases", {}).items():
        if phase_name == "integration":
            continue
        if isinstance(phase_data, dict) and "status" in phase_data:
            print(f"  {phase_name}: {phase_data['status']}")


def cmd_report(config_path: str | None = None) -> None:
    """Phase 8: Generate the self-contained HTML report from real pipeline data."""
    import subprocess
    import sys

    root = Path(__file__).parent.parent.parent
    cmd = [sys.executable, "-m", "scripts.generate_report"]
    if _DOMAIN:
        cmd += ["--domain", _DOMAIN]
    subprocess.run(
        cmd,
        cwd=root,
        check=True,
    )


def cmd_all(config_path: str | None = None) -> None:
    """Run all 7 phases sequentially."""
    steps = [
        ("Phase 1: Collect vacancies", cmd_collect),
        ("Phase 2: Classify STEM/IT", cmd_classify),
        ("Phase 2.5: Extract vacancy characteristics", cmd_phase25),
        ("Phase 3: Discover roles", cmd_roles),
        ("Phase 4: Extract skills", cmd_skills),
        ("Phase 5: Map to characteristics", cmd_characteristics),
        ("Phase 5b: Soft competences", cmd_soft),
        ("Phase 6: Build models", cmd_model),
        ("Phase 7: Validate", cmd_validate),
        ("Phase 8: Generate HTML report", cmd_report),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        try:
            fn(config_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 0: Historical data collection
# ---------------------------------------------------------------------------


def cmd_phase0_census(config_path: str | None = None) -> None:
    """Phase 0: Run Wayback CDX coverage census across all sources."""
    from krm.phase0.cdx import CdxQuerier
    from krm.phase0.census import CensusRunner

    cfg = _get_config(config_path)
    querier = CdxQuerier(
        endpoint=cfg.phase0_cdx_endpoint,
        rate_limit_rps=cfg.phase0_cdx_rate_limit_rps,
    )
    runner = CensusRunner(querier)
    report = runner.run_all()

    # Write JSON report
    report_dir = Path(cfg.phase0_census_cache_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_json(report_dir / "census_report.json")

    print(report.to_table())


def cmd_phase0_hhru(config_path: str | None = None) -> None:
    """Phase 0: Collect hh.ru historical vacancies from Wayback Machine."""
    from krm.phase0.collectors.hhru_wayback import collect_hhru_wayback

    cfg = _get_config(config_path)
    years = cfg.phase0_hhru_wayback_years
    total = 0
    for year in range(years[0], years[1] + 1):
        try:
            stored = collect_hhru_wayback(year, cfg)
            total += stored
            print(f"  hh.ru {year}: {stored} records stored")
        except ValueError as e:
            print(f"  hh.ru {year}: SKIPPED ({e})")
    print(f"Phase 0 hh.ru complete: {total} total records stored")


def cmd_phase0_linkedin(config_path: str | None = None) -> None:
    """Phase 0: Collect LinkedIn historical job postings from Wayback Machine."""
    from krm.phase0.collectors.linkedin_wayback import collect_linkedin_wayback

    cfg = _get_config(config_path)
    total = 0
    for year in range(2013, 2027):
        try:
            stored = collect_linkedin_wayback(year, cfg)
            total += stored
            print(f"  LinkedIn {year}: {stored} records stored")
        except ValueError as e:
            print(f"  LinkedIn {year}: SKIPPED ({e})")
    print(f"Phase 0 LinkedIn complete: {total} total records stored")


def cmd_phase0_trudvsem(config_path: str | None = None) -> None:
    """Phase 0: Collect Trudvsem open data vacancies (hh.ru sourced)."""
    from krm.phase0.collectors.trudvsem import TrudvsemCollector
    from krm.phase0.schema import normalize_record
    from krm.phase0.storage import (
        finish_phase0_run,
        get_phase0_connection,
        init_phase0_tables,
        start_phase0_run,
        upsert_phase0_vacancy,
    )

    cfg = _get_config(config_path)
    if not cfg.phase0_trudvsem_enabled:
        print("Trudvsem collector disabled")
        return

    run_id = "trudvsem"
    conn = get_phase0_connection(cfg.phase0_db_path)
    init_phase0_tables(conn)
    start_phase0_run(conn, run_id, "trudvsem", cfg.phase0_trudvsem_base_url)

    collector = TrudvsemCollector(
        base_url=cfg.phase0_trudvsem_base_url,
        rate_limit_rps=cfg.phase0_trudvsem_rate_limit_rps,
    )
    try:
        vacancies = collector.collect_all(
            date_from=cfg.phase0_trudvsem_date_from,
            date_to=cfg.phase0_trudvsem_date_to,
        )
        stored = 0
        for v in vacancies:
            normalized = normalize_record(v)
            upsert_phase0_vacancy(conn, run_id, normalized["id"], normalized)
            stored += 1

        finish_phase0_run(conn, run_id, len(vacancies), stored)
        print(f"Trudvsem: {stored} records stored")
    finally:
        collector.close()
        conn.close()


# ---------------------------------------------------------------------------
# Trudvsem keyword-based collection (100k+ scale runner)
# ---------------------------------------------------------------------------

# Top Russian regions by economic activity for geo-partitioning.
# Used when a keyword exceeds the 10k offset cap.
_RUSSIAN_REGION_CODES: list[str] = [
    "7700000000000",  # Москва
    "5000000000000",  # Московская область
    "7800000000000",  # Санкт-Петербург
    "4700000000000",  # Ленинградская область
    "6600000000000",  # Свердловская область
    "5400000000000",  # Новосибирская область
    "1600000000000",  # Республика Татарстан
    "2300000000000",  # Краснодарский край
    "7400000000000",  # Челябинская область
    "5200000000000",  # Нижегородская область
    "6300000000000",  # Самарская область
    "0200000000000",  # Республика Башкортостан
    "6100000000000",  # Ростовская область
    "5900000000000",  # Пермский край
    "5500000000000",  # Омская область
    "3800000000000",  # Иркутская область
    "2400000000000",  # Красноярский край
    "8600000000000",  # Ханты-Мансийский АО
    "3600000000000",  # Воронежская область
    "3400000000000",  # Волгоградская область
    "4200000000000",  # Кемеровская область
    "5600000000000",  # Оренбургская область
    "2200000000000",  # Алтайский край
    "6400000000000",  # Саратовская область
    "7300000000000",  # Ульяновская область
    "1800000000000",  # Удмуртская Республика
    "2700000000000",  # Хабаровский край
    "2500000000000",  # Приморский край
    "7200000000000",  # Тюменская область
    "3700000000000",  # Ивановская область
    "6200000000000",  # Рязанская область
    "3300000000000",  # Владимирская область
    "4000000000000",  # Калужская область
    "4600000000000",  # Курская область
    "3200000000000",  # Брянская область
    "7100000000000",  # Тульская область
    "7600000000000",  # Ярославская область
    "2900000000000",  # Архангельская область
    "3500000000000",  # Вологодская область
    "3900000000000",  # Калининградская область
    "1000000000000",  # Республика Карелия
    "1100000000000",  # Республика Коми
    "2600000000000",  # Ставропольский край
    "3000000000000",  # Астраханская область
    "6800000000000",  # Тамбовская область
    "5800000000000",  # Пензенская область
    "2100000000000",  # Чувашская Республика
    "1200000000000",  # Республика Марий Эл
    "1300000000000",  # Республика Мордовия
    "7500000000000",  # Забайкальский край
    "2800000000000",  # Амурская область
    "6500000000000",  # Сахалинская область
    "4100000000000",  # Камчатский край
    "4900000000000",  # Магаданская область
    "1400000000000",  # Республика Саха (Якутия)
    "8700000000000",  # Чукотский АО
    "8300000000000",  # Ненецкий АО
    "8900000000000",  # Ямало-Ненецкий АО
    "7900000000000",  # Еврейская АО
]


def cmd_trudvsem_collect(config_path: str | None = None) -> None:
    """Collect STEM vacancies from Trudvsem API by keyword search.

    Uses text-based keyword search with region-code partitioning for
    keywords exceeding the API's ~10,000 offset cap.  Stores results
    directly in the main pipeline DuckDB (data/krm.duckdb).

    Usage:
        python -m krm.cli trudvsem-collect           # full collection
        python -m krm.cli trudvsem-collect --dry-run  # census only
    """
    import json
    import sys
    from datetime import datetime, timezone

    from krm.lib.io import get_connection, init_tables, upsert_raw_vacancy
    from krm.phase0.collectors.trudvsem import (
        MAX_OFFSET,
        TrudvsemCollector,
    )

    dry_run = "--dry-run" in sys.argv

    cfg = _get_config(config_path)
    keywords: list[str] = list(cfg.keywords)

    # Expanded STEM keyword list for broad coverage
    extra_keywords = [
        "инженер-исследователь", "научный работник", "преподаватель физики",
        "преподаватель химии", "преподаватель биологии",
        "инженер-химик", "инженер-физик", "инженер-биолог",
        "исследователь", "научный сотрудник лаборатории",
        "химик-аналитик", "химик-технолог", "биолог-исследователь",
        "физик-экспериментатор", "физик-теоретик",
        "лаборант химического анализа", "лаборант-исследователь",
        "специалист по контролю качества", "инженер-метролог",
        "инженер по стандартизации", "инженер-конструктор",
        "инженер-проектировщик", "инженер-схемотехник",
        "радиофизик", "радиохимик", "ядерная физика",
        "нанотехнолог", "биотехнолог", "биоинженер",
        "агроном-исследователь", "почвовед", "геофизик",
        "геохимик", "гидролог", "гляциолог", "сейсмолог",
        "специалист по неразрушающему контролю", "дефектоскопист",
        "оптик", "лазерная физика", "спектроскопист",
        "хроматографист", "масс-спектрометрист",
        "электронный микроскопист", "рентгеноструктурный анализ",
        "материаловед-исследователь", "инженер-материаловед",
        "физик-ядерщик", "физик-ускорительщик",
        "криогенная техника", "вакуумная техника",
        "квантовая физика", "физика плазмы",
        "аэродинамика", "гидродинамика",
        "сопротивление материалов", "теоретическая механика",
        "вычислительная математика", "прикладная математика",
        "молекулярная биология", "клеточная биология",
        "иммунология", "вирусология", "бактериология",
        "энзимология", "биоорганическая химия",
        "медицинская химия", "фармацевтическая химия",
        "химия полимеров", "физическая химия",
        "электрохимия", "фотохимия", "радиационная химия",
        "агрохимия", "геохимия", "космохимия",
        "аналитическая химия", "органическая химия",
        "неорганическая химия", "коллоидная химия",
        "экологический мониторинг", "охрана окружающей среды",
        "климатология", "океанография", "вулканология",
        "петрография", "минералогия", "палеонтология",
        "стратиграфия", "тектоника", "геоморфология",
        "физика атмосферы", "физика океана",
        "космические исследования", "астрофизика",
        "радиоастрономия", "планетология",
    ]
    all_keywords = list(dict.fromkeys(keywords + extra_keywords))  # dedup preserve order

    conn = get_connection()
    init_tables(conn)
    run_id = f"trudvsem-kw_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    collector = TrudvsemCollector(
        rate_limit_rps=cfg.phase0_trudvsem_rate_limit_rps,
    )

    total_stored = 0
    keyword_counts: dict[str, int] = {}

    try:
        for kw in all_keywords:
            print(f"\n{'=' * 50}")
            print(f"Keyword: {kw}")

            api_count = collector.count(text=kw)
            print(f"  API total: {api_count}")

            if api_count == 0:
                keyword_counts[kw] = 0
                continue

            if dry_run:
                keyword_counts[kw] = min(api_count, MAX_OFFSET)
                continue

            effective_limit = min(api_count, MAX_OFFSET)

            if api_count > MAX_OFFSET:
                print(f"  Exceeds offset cap ({MAX_OFFSET}) — partitioning by region")
                stored_kw = _collect_with_region_partitioning(
                    collector, conn, run_id, kw, effective_limit
                )
            else:
                stored_kw = _collect_keyword_full(collector, conn, run_id, kw)

            keyword_counts[kw] = stored_kw
            total_stored += stored_kw
            print(f"  Stored: {stored_kw} (total: {total_stored})")

    finally:
        collector.close()
        conn.close()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Collection complete: {total_stored} records from {len(all_keywords)} keywords")
    print(f"Run ID: {run_id}")
    if dry_run:
        estimated = sum(keyword_counts.values())
        print(f"DRY RUN — estimated: {estimated} records")


def _collect_keyword_full(
    collector: Any, conn: Any, run_id: str, keyword: str
) -> int:
    """Collect all available results for a single keyword."""
    from krm.phase0.collectors.trudvsem import MAX_OFFSET
    from krm.lib.io import upsert_raw_vacancy

    vacancies = collector.collect_all(text=keyword)
    stored = 0
    for v in vacancies:
        upsert_raw_vacancy(conn, run_id, v["id"], v)
        stored += 1
    return stored


def _collect_with_region_partitioning(
    collector: Any, conn: Any, run_id: str, keyword: str, max_total: int
) -> int:
    """Collect vacancies for a keyword using region-code partitioning.

    When a keyword exceeds the 10k offset cap, collect separately for
    each major Russian region to stay under the limit per query.
    """
    from krm.lib.io import upsert_raw_vacancy

    stored = 0
    seen: set[str] = set()

    for rc in _RUSSIAN_REGION_CODES:
        region_count = collector.count(text=keyword, region_code=rc)
        if region_count == 0:
            continue

        print(f"    region {rc}: {region_count}")
        vacancies = collector.collect_all(text=keyword, region_code=rc)

        for v in vacancies:
            vid = v["id"]
            if vid not in seen:
                upsert_raw_vacancy(conn, run_id, vid, v)
                seen.add(vid)
                stored += 1

        if region_count <= 100:
            continue  # small region, continue to next

    return stored


def cmd_phase0_rostud(config_path: str | None = None) -> None:
    """Phase 0: Collect Rostrud bulk CSV/XLSX vacancy data."""
    from krm.phase0.collectors.rostud import collect_rostud

    cfg = _get_config(config_path)
    stored = collect_rostud(cfg)
    print(f"Rostrud: {stored} records stored")


def cmd_phase0_all(config_path: str | None = None) -> None:
    """Phase 0: Run all historical data collectors sequentially."""
    steps: list[tuple[str, object]] = [
        ("Phase 0: Census (coverage estimate)", cmd_phase0_census),
        ("Phase 0: hh.ru Wayback", cmd_phase0_hhru),
        ("Phase 0: LinkedIn Wayback", cmd_phase0_linkedin),
        ("Phase 0: Trudvsem", cmd_phase0_trudvsem),
        ("Phase 0: Rostrud", cmd_phase0_rostud),
    ]
    for label, fn in steps:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        try:
            fn(config_path)  # type: ignore[operator]
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)


def cmd_phase0_verify(config_path: str | None = None) -> None:
    """Phase 0: Verify collection run summaries from the Phase 0 database."""
    from krm.phase0.storage import get_phase0_connection

    cfg = _get_config(config_path)
    conn = get_phase0_connection(cfg.phase0_db_path)
    try:
        rows = conn.execute(
            (
                "SELECT run_id, source, source_url_pattern, started_at, "
                "completed_at, records_fetched, records_stored "
                "FROM phase0_runs ORDER BY started_at"
            )
        ).fetchall()
        if not rows:
            print("No Phase 0 runs found.")
            return
        print(f"{'Run ID':<30} {'Source':<18} {'Started':<20} {'Completed':<20} {'Stored':>8}")
        print("-" * 100)
        for row in rows:
            started = row[3] or "—"
            completed = row[4] or "—"
            print(
                f"{row[0]:<30} {row[1]:<18} {started:<20} {completed:<20} {row[6]:>8}"
            )
        total_stored = conn.execute(
            "SELECT SUM(records_stored) FROM phase0_runs"
        ).fetchone()[0]
        print(f"\nTotal records across all runs: {total_stored or 0}")
    finally:
        conn.close()


def cmd_phase0_merge(config_path: str | None = None) -> None:
    """Phase 0: Merge Phase 0 historical database into the main pipeline database."""
    from krm.phase0.storage import merge_into_main

    cfg = _get_config(config_path)
    n = merge_into_main(cfg.phase0_db_path)
    print(f"Merged {n} records from Phase 0 into main database")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: krm <command> [--config config.yaml]")
        cmds = (
            "Commands: collect, classify, phase25, roles, skills, extract-skills, "
            "characteristics, soft, model, validate, report, all, "
            "phase0-census, phase0-hhru, phase0-linkedin, phase0-trudvsem, "
            "phase0-rostud, phase0-all, phase0-verify, phase0-merge"
        )
        print(cmds)
        sys.exit(1)

    from krm.lib.logging import setup_logging

    setup_logging()

    command = sys.argv[1]
    config_path: str | None = None

    # Parse --config and --domain flags
    global _DOMAIN
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
        if arg == "--domain" and i + 1 < len(args):
            _DOMAIN = args[i + 1]

    commands = COMMANDS

    if command not in commands:
        print(f"Unknown command: {command}")
        sys.exit(1)

    fn = commands[command]
    if fn is not None and hasattr(fn, "__call__"):
        fn(config_path)  # type: ignore[operator]


def cmd_trudvsem_region_scan(config_path: str | None = None) -> None:
    """Collect ALL vacancies by region (no text filter) for maximum coverage.

    Region-only browsing has full pagination support (up to ~10,000 per region),
    unlike text search which caps at ~110 results. Post-filtering for STEM
    happens in Phase 2 classification.

    Usage:
        python -m krm.cli trudvsem-region-scan
        python -m krm.cli trudvsem-region-scan --regions=77,78,50  # specific regions
        python -m krm.cli trudvsem-region-scan --dry-run
    """
    import json
    import sys
    from datetime import datetime, timezone

    from krm.lib.io import get_connection, init_tables, upsert_raw_vacancy
    from krm.phase0.collectors.trudvsem import (
        MAX_OFFSET,
        FIRST_PAGE_LIMIT,
        NEXT_PAGE_LIMIT,
        TrudvsemCollector,
    )

    dry_run = "--dry-run" in sys.argv

    # Parse optional --regions= flag
    regions_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--regions=")), None)
    if regions_arg:
        target_regions = [rc.strip() for rc in regions_arg.split(",") if rc.strip()]
    else:
        target_regions = list(_RUSSIAN_REGION_CODES)

    cfg = _get_config(config_path)
    conn = get_connection()
    init_tables(conn)
    run_id = f"trudvsem-region_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    collector = TrudvsemCollector(rate_limit_rps=cfg.phase0_trudvsem_rate_limit_rps)

    total_stored = 0

    try:
        for rc in target_regions:
            # Count without text filter
            region_total = collector.count(region_code=rc)
            if region_total == 0:
                print(f"Region {rc}: 0 vacancies, skip")
                continue

            effective = min(region_total, MAX_OFFSET)
            print(f"\nRegion {rc}: {region_total} total, collecting up to {effective}")

            if dry_run:
                total_stored += effective
                continue

            # Collect region-only (no text search) with full pagination
            stored = 0
            vacancies = collector.collect_all(region_code=rc)
            for v in vacancies:
                upsert_raw_vacancy(conn, run_id, v["id"], v)
                stored += 1

            total_stored += stored
            print(f"  Stored: {stored}")

    finally:
        collector.close()
        conn.close()

    print(f"\n{'=' * 60}")
    print(f"Region scan complete: {total_stored} records from {len(target_regions)} regions")
    print(f"Run ID: {run_id}")
    if dry_run:
        print(f"DRY RUN — estimated: {total_stored} records")


COMMANDS: dict[str, object] = {
    "collect": cmd_collect,
    "classify": cmd_classify,
    "phase25": cmd_phase25,
    "roles": cmd_roles,
    "skills": cmd_skills,
    "characteristics": cmd_characteristics,
    "soft": cmd_soft,
    "model": cmd_model,
    "validate": cmd_validate,
    "report": cmd_report,
    "all": cmd_all,
    "phase0-census": cmd_phase0_census,
    "phase0-hhru": cmd_phase0_hhru,
    "phase0-linkedin": cmd_phase0_linkedin,
    "phase0-trudvsem": cmd_phase0_trudvsem,
    "phase0-rostud": cmd_phase0_rostud,
    "phase0-all": cmd_phase0_all,
    "phase0-verify": cmd_phase0_verify,
    "phase0-merge": cmd_phase0_merge,
    "extract-skills": cmd_extract_skills,
    "trudvsem-collect": cmd_trudvsem_collect,
    "trudvsem-region-scan": cmd_trudvsem_region_scan,
    "collect-domain": cmd_collect_domain,
}


if __name__ == "__main__":
    main()
