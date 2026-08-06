"""Data ingestion: import vacancies from external datasets and historical archives.

HH.ru live scraping is one data source. This package adds support for:
- CSV/JSONL dumps (common export format for pre-scraped data)
- HuggingFace datasets (trewwxsav/IT_vacancies_from_hh.ru, etc.)
- Trudvsem.ru bulk CSV (government job portal, no auth required)
- Wayback Machine JSON captures of HH.ru search pages

All sources normalize to VacancyData and feed into the same DuckDB pipeline.
"""

from hh_competency.data.importer import DataImporter, DataSource, ImportResult

__all__ = ["DataImporter", "DataSource", "ImportResult"]
