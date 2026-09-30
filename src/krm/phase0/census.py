"""Phase 0 Census — CDX API coverage estimates before full extraction.

Estimates dataset sizes per source, per year to inform extraction priority and
resource allocation.  Queries the Wayback Machine CDX API for each source/year
combination and reports capture counts and unique URL counts.

Use :class:`CensusRunner` with a :class:`~krm.phase0.cdx.CdxQuerier` to
produce a :class:`CensusReport`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from krm.phase0.cdx import CdxQuerier


@dataclass(frozen=True, slots=True)
class CensusResult:
    """A single census row: one source × year × URL pattern.

    Each result represents the estimated dataset size for a specific
    Wayback Machine capture source, URL pattern, and year.

    Attributes:
        source:  Source label (``"wayback-hhru"``, ``"wayback-linkedin"``, …).
        url_pattern:  CDX URL pattern queried.
        year:  Calendar year of the capture window.
        capture_count:  Total number of captures (one CDX page estimate).
        unique_urls:  Number of unique URLs after ``collapse:urlkey`` dedup.
    """

    source: str
    url_pattern: str
    year: int
    capture_count: int
    unique_urls: int


@dataclass(frozen=True, slots=True)
class CensusReport:
    """Aggregated census report for all queried sources.

    Attributes:
        generated_at:  ISO-8601 UTC timestamp when the report was generated.
        results:  Individual :class:`CensusResult` rows per source/year.
        total_estimated_records:  Sum of all ``capture_count`` values.
    """

    generated_at: str
    results: list[CensusResult] = field(default_factory=list)
    total_estimated_records: int = 0

    def to_json(self, path: Path) -> None:
        """Write the report as a JSON file.

        Creates parent directories if they do not exist.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {
            "generated_at": self.generated_at,
            "total_estimated_records": self.total_estimated_records,
            "results": [
                {
                    "source": r.source,
                    "url_pattern": r.url_pattern,
                    "year": r.year,
                    "capture_count": r.capture_count,
                    "unique_urls": r.unique_urls,
                }
                for r in self.results
            ],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def to_table(self) -> str:
        """Return a human-readable ASCII table of the census results."""
        lines: list[str] = []
        lines.append("Census Report")
        lines.append(f"Generated: {self.generated_at}")
        lines.append(f"Total estimated records: {self.total_estimated_records:,}")
        lines.append("")

        header = (
            f"{'Source':<24} {'Pattern':<36} {'Year':>6} "
            f"{'Captures':>10} {'Unique':>8}"
        )
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)

        for r in self.results:
            line = (
                f"{r.source:<24} {r.url_pattern:<36} {r.year:>6}"
                + f"  {r.capture_count:>10,} {r.unique_urls:>8,}"
            )
            lines.append(line)

        return "\n".join(lines)


class CensusRunner:
    """Runs CDX census queries across all configured sources and years.

    Args:
        querier:  A :class:`~krm.phase0.cdx.CdxQuerier` instance used to
            issue CDX API queries.  For testing, pass a mock.
    """

    def __init__(self, querier: CdxQuerier) -> None:
        self._querier: CdxQuerier = querier

    # ------------------------------------------------------------------
    # Per-source runners
    # ------------------------------------------------------------------

    def run_hhru_wayback(self) -> list[CensusResult]:
        """Census hh.ru Wayback Machine captures covering 2010-2025.

        Queries a legacy ``hh.ru/vacancy*.do`` pattern for 2010-2011 and a
        modern ``hh.ru/vacancy/*`` pattern for every year in 2010-2025.
        """
        results: list[CensusResult] = []

        # Legacy pattern: hh.ru/vacancy*.do (2010-2011)
        for year in range(2010, 2012):
            results.append(
                self._census_year(
                    source="wayback-hhru",
                    url_pattern="hh.ru/vacancy*.do",
                    year=year,
                )
            )

        # Modern pattern: hh.ru/vacancy/* (2010-2025)
        for year in range(2010, 2026):
            results.append(
                self._census_year(
                    source="wayback-hhru",
                    url_pattern="hh.ru/vacancy/*",
                    year=year,
                )
            )

        return results

    def run_linkedin_wayback(self) -> list[CensusResult]:
        """Census LinkedIn job posting Wayback Machine captures.

        Queries the legacy ``linkedin.com/jobs2/view/*`` pattern for
        2013-2016 and the modern ``linkedin.com/jobs/view/*`` pattern for
        2019-2026.
        """
        results: list[CensusResult] = []

        # Legacy pattern: linkedin.com/jobs2/view/* (2013-2016)
        for year in range(2013, 2017):
            results.append(
                self._census_year(
                    source="wayback-linkedin",
                    url_pattern="linkedin.com/jobs2/view/*",
                    year=year,
                )
            )

        # Modern pattern: linkedin.com/jobs/view/* (2019-2026)
        for year in range(2019, 2027):
            results.append(
                self._census_year(
                    source="wayback-linkedin",
                    url_pattern="linkedin.com/jobs/view/*",
                    year=year,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def run_all(self) -> CensusReport:
        """Run census for all sources and return an aggregated report."""
        results: list[CensusResult] = []
        results.extend(self.run_hhru_wayback())
        results.extend(self.run_linkedin_wayback())

        total = sum(r.capture_count for r in results)

        return CensusReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            results=results,
            total_estimated_records=total,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _census_year(
        self, source: str, url_pattern: str, year: int,
    ) -> CensusResult:
        """Issue both collapsed and un-collapsed CDX queries for one year.

        Two queries per year:
        * Without collapse → capture count (total snapshots).
        * With ``collapse=urlkey`` → unique URL count.
        """
        from_ts = f"{year}0101"
        to_ts = f"{year}1231"

        # Total captures (one page, default CDX limit).
        captures = self._querier.query(
            url_pattern=url_pattern,
            from_ts=from_ts,
            to_ts=to_ts,
            status_filter=200,
        )
        capture_count = len(captures)

        # Unique URLs after collapse:urlkey dedup.
        unique = self._querier.query(
            url_pattern=url_pattern,
            from_ts=from_ts,
            to_ts=to_ts,
            collapse="urlkey",
            status_filter=200,
        )
        unique_urls = len(unique)

        return CensusResult(
            source=source,
            url_pattern=url_pattern,
            year=year,
            capture_count=capture_count,
            unique_urls=unique_urls,
        )
