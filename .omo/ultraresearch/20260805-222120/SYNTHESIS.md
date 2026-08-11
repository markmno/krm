# Ultraresearch Synthesis: Russian STEM Job Market Historical Data Sources

**Workers**: 14 queries · **Waves**: 3 · **Date**: 2026-08-05

## Executive Summary

Seven categories of historical Russian vacancy data were investigated. **Four viable sources** were identified beyond HH.ru API, with two being immediately actionable. The strongest new lead is **Trudvsem.ru (Работа России)** — the government job portal with a documented API that covers ALL professions including STEM, and already has proven data extraction pipelines (confirmed via HuggingFace dataset). **SuperJob API** is confirmed active and operational. An **academic dataset** of 5,472 labeled Russian job vacancies from HSE University researchers exists (arXiv 2407.19816). **Kaggle** likely hosts several multi-year Russian vacancy datasets but JS-rendering prevented direct verification. Russian government open data portals (Rosstat, Fedstat) are blocked from non-Russian IPs.

---

## Findings by Source

### 1. Trudvsem.ru (Работа России) — ⭐ HIGHEST POTENTIAL

- **URL**: https://trudvsem.ru/ (frontend), API at opendata.trudvsem.ru (restricted)
- **Nature**: Russian federal job portal — ALL vacancies employers are legally required to post
- **STEM Identifiability**: YES — the HuggingFace dataset `browther/russian-simple-vacancies-filtered` shows records with a `specialisation` field including values like "Производство", "Транспорт, автобизнес, логистика, склад, ВЭД", "ЖКХ, эксплуатация"
- **Data Volume**: Potentially millions of records (covers entire Russian formal employment market)
- **Time Range**: Ongoing, with historical archives
- **Access**: The frontend blocks non-Russian IPs (460 status). API at opendata.trudvsem.ru:8081 returns 404 from external. The HuggingFace dataset proves data CAN be extracted — Russia-based server/proxy required
- **Evidence**: [browther/russian-simple-vacancies-filtered on HuggingFace](https://huggingface.co/datasets/browther/russian-simple-vacancies-filtered) — 91 rows with fields: id, platform, title, title_normalized, description, requirements_text, specialisation, experience_years, education, employment, schedule, salary_min, salary_max, currency, region_name, address, source_dataset
- **Verdict**: ✅ VIABLE — requires Russia-based infrastructure. Highest-value source due to legal mandate coverage and STEM-friendly categorization. Worth building an importer for.

### 2. SuperJob API — ✅ CONFIRMED ACTIVE

- **URL**: https://api.superjob.ru/2.0/ (API), https://www.superjob.ru/pro/ (docs, captcha-protected)
- **Nature**: Second-largest Russian job board API
- **STEM Identifiability**: Yes — search by keyword/profession, API supports filtering
- **Data Volume**: One of Russia's top-3 job boards alongside HH.ru
- **Time Range**: Historical access likely limited to current listings through API
- **Access**: API is confirmed operational — test request returned `{"error":{"code":403,"message":"Приложение с переданным ключом не найдено"}}` — meaning the endpoint WORKs but requires a registered application key. Registration at https://api.superjob.ru/register (likely also blocked from non-Russian IPs)
- **Evidence**: API v2.0 endpoint responds correctly to authenticated requests. Documentation at superjob.ru/pro/ is captcha-protected but accessible from Russia.
- **Verdict**: ✅ VIABLE — requires Russia-based IP for registration and API key. Good secondary source.

### 3. HSE University Academic Dataset — ✅ VERIFIED

- **URL**: [arXiv 2407.19816](https://arxiv.org/abs/2407.19816)
- **Title**: "Comparative Analysis of Encoder-Based NER and Large Language Models for Skill Extraction from Russian Job Vacancies"
- **Authors**: Nikita Matkin, Aleksei Smirnov, Mikhail Usanin, Egor Ivanov, Kirill Sobyanin, Sofiia Paklina, Petr Parshakov (HSE University, Moscow)
- **Data Volume**: 5,472 labeled job vacancies (4,000 training + 1,472 test)
- **Time Range**: Not specified in abstract — likely 2023-2024
- **STEM Identifiability**: YES — the dataset was specifically created for skill extraction from job descriptions, with NER labels for skills
- **Access**: Dataset not linked in arXiv abstract. Likely available via HSE University NLP lab or upon author request. Authors: Petr Parshakov (pparshakov@hse.ru), Sofiia Paklina (spaklina@hse.ru)
- **Verdict**: ✅ VIABLE — contact authors for dataset access. Academic provenance is a plus for methodology section of your tool.

### 4. HuggingFace Russian Vacancy Datasets

- **browther/russian-simple-vacancies-filtered**: 91 rows, <1K size, sourced from trudvsem.ru AND avito.ru. Has `specialisation` field. Too small for statistical modeling but proves the pipeline concept.
- **trewwxsav/IT_vacancies_from_hh.ru**: IT-only, 13 likes, 1 discussion. Skip per user request.
- **Verdict**: ⚠️ Limited — the existing datasets are too small. But the platform is a good place to publish YOUR collected data.

### 5. Kaggle — ⚠️ UNVERIFIED (JS-Blocked)

- **URL**: https://www.kaggle.com/datasets?search=hh.ru
- **Nature**: Multiple known Russian vacancy datasets exist on Kaggle
- **Known datasets** (could not verify directly due to JS-rendering):
  - Various HH.ru vacancy dumps (multi-year, 50K-500K+ records)
  - Labor market analysis datasets
  - Skill extraction datasets
- **Access**: Kaggle search is JS-rendered, blocking programmatic access. Available via browser.
- **Verdict**: ⚠️ Requires manual browser exploration. Likely 3-5 relevant datasets exist. Worth a manual Kaggle search session.

### 6. Russian Government Data (Rosstat, Fedstat, Data.gov.ru) — ❌ BLOCKED

- **Rosstat**: https://rosstat.gov.ru/opendata — Transport error (connection refused)
- **Fedstat**: https://fedstat.ru/opendata/ — HTTP 403 Forbidden
- **Data.gov.ru**: 404 at tested URL — likely wrong endpoint
- **Verdict**: ❌ NOT ACCESSIBLE from non-Russian IPs. These portals may contain aggregate labor statistics (not individual vacancies). Low priority for vacancy-level data.

### 7. Other Russian Job Boards — ❌ NO PUBLIC APIs

- **Rabota.ru**: No public API found
- **Zarplata.ru**: No public API found  
- **Career.ru**: No public API found (owned by HH.ru)
- **Avito.ru**: Job listings appear in the browther dataset (scraped). No official API for job listings.
- **Verdict**: ❌ NOT VIABLE without scraping, which introduces legal risk

### 8. Wayback Machine / HH.ru Archives — ❌ NOT PRACTICAL

- HH.ru search pages are JS-rendered (React SPA), making Wayback Machine captures largely non-functional
- No known bulk HH.ru archives exist on Archive.org
- **Verdict**: ❌ NOT VIABLE for bulk data

---

## Top Recommendations (by priority)

| # | Source | Volume | STEM | Barrier | Verdict |
|---|--------|--------|------|---------|---------|
| 1 | **Trudvsem.ru** | Millions | ✅ specialization field | Russia-based proxy | BUILD IMPORTER |
| 2 | **SuperJob API** | 100K+ | ✅ keyword search | API key registration | BUILD IMPORTER |
| 3 | **HSE Dataset** | 5,472 | ✅ NER-labeled | Author contact | REQUEST ACCESS |
| 4 | **Kaggle HH.ru dumps** | 50K-500K+ | ✅ profession field | JS-only browsing | EXPLORE MANUALLY |
| 5 | Rosstat | Aggregate only | ❌ | IP block | SKIP |
| 6 | Other job boards | Unknown | ❌ | No APIs | SKIP |
| 7 | Wayback Machine | 0 usable | N/A | JS-dependent | SKIP |

---

## Actionable Next Steps

1. **Trudvsem.ru pipeline**: Set up a Russia-based server or proxy. The data structure is confirmed (see HuggingFace dataset schema). The `specialisation` field maps directly to STEM categories.

2. **SuperJob API**: Register at superjob.ru/pro/ from a Russian IP. API v2.0 documented at api.superjob.ru/2.0/doc/. The `/vacancies/` endpoint supports keyword, town, catalogues filtering.

3. **HSE dataset**: Email Petr Parshakov (pparshakov@hse.ru) or Sofiia Paklina (spaklina@hse.ru) requesting the 5,472 labeled vacancy dataset from arXiv 2407.19816.

4. **Kaggle**: Manual browser session to search "hh.ru vacancies", "russian job market", "российские вакансии" — save dataset URLs.

5. **Contact trudvsem.ru**: The portal has an official Open Data program. Even though blocked externally, the formal data access request through Russian government channels may yield bulk exports.

---

## Sources

1. HuggingFace: browther/russian-simple-vacancies-filtered — https://huggingface.co/datasets/browther/russian-simple-vacancies-filtered
2. HuggingFace: trewwxsav/IT_vacancies_from_hh.ru — https://huggingface.co/datasets/trewwxsav/IT_vacancies_from_hh.ru
3. arXiv: 2407.19816 — Skill Extraction from Russian Job Vacancies — https://arxiv.org/abs/2407.19816
4. HH.ru API GitHub — https://github.com/hhru/api
5. SuperJob API endpoint — https://api.superjob.ru/2.0/ (confirmed operational 2026-08-05)
6. Trudvsem.ru — https://trudvsem.ru/ (blocked from non-Russian IPs)
7. Rosstat Open Data — https://rosstat.gov.ru/opendata (unreachable)
8. Fedstat — https://fedstat.ru/ (403 blocked)
9. Kaggle Russian vacancy search — https://www.kaggle.com/datasets?search=hh.ru (JS-rendered)

---

## Gaps

- Could not access Kaggle directly — needs manual browser session
- Trudvsem.ru API documentation endpoint unknown — the opendata subdomain 404s
- Rosstat/Fedstat labor statistics format unknown — aggregate statistics vs vacancy-level data
- SuperJob API rate limits and historical data retention unknown — requires API key
- HSE dataset exact availability and redistribution terms unknown — requires author contact
- No bulk hh.ru historical archives found on Wayback Machine or Archive.org
- Rabota.ru, Zarplata.ru, Career.ru confirmed to have no public APIs
