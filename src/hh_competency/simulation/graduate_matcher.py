"""Graduate-to-market role matching simulation.

Answers: "Which ITMO graduates match which real market roles?"
Simulates graduate curricula against specialty_profiles role data to compute
market-ship readiness scores using Jaccard similarity.

SIZE_OK: Indivisible simulation engine — the main simulation function, its
data types, role-loading helpers, and Rich output formatter form a single
cohesive unit whose callers should see as one import.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from hh_competency.storage.db import Database


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


class GraduateCurriculum:
    """Skills a graduate has from their curriculum or configuration."""

    __slots__ = ("specialty_name", "skills", "source")

    def __init__(
        self,
        specialty_name: str,
        skills: list[str],
        source: str = "config",
    ) -> None:
        self.specialty_name = specialty_name
        self.skills = skills
        self.source = source

    @property
    def skill_set(self) -> set[str]:
        """Lowercased skill lemmas for fast set operations."""
        return {s.lower() for s in self.skills}


class GraduatePool:
    """Collection of graduates with different specializations."""

    def __init__(self) -> None:
        self._graduates: list[GraduateCurriculum] = []

    @property
    def graduates(self) -> list[GraduateCurriculum]:
        return list(self._graduates)

    def add_curriculum(self, graduate: GraduateCurriculum) -> None:
        """Add a single graduate curriculum to the pool."""
        self._graduates.append(graduate)

    @classmethod
    def from_specialty_config(
        cls, specialties_config: dict[str, Any]
    ) -> GraduatePool:
        """Build graduate pool from specialties.yml NLP keywords as curricula.

        Each specialty's ``nlp_keywords`` becomes a pretend curriculum for a
        graduate specializing in that field.  This provides a baseline until
        real ITMO curriculum data is available.

        Args:
            specialties_config: The ``"specialties"`` dictionary from a YAML
                config (e.g. ``config["specialties"]``).

        Returns:
            A ``GraduatePool`` with one ``GraduateCurriculum`` per specialty.
        """
        pool = cls()
        for name, spec in specialties_config.items():
            nl_keywords = spec.get("nlp_keywords", [])
            if nl_keywords:
                pool.add_curriculum(
                    GraduateCurriculum(
                        specialty_name=name,
                        skills=list(nl_keywords),
                        source="config",
                    )
                )
        return pool

    def __len__(self) -> int:
        return len(self._graduates)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._graduates)


# ------------------------------------------------------------------
# Role loading helpers
# ------------------------------------------------------------------


def _load_all_role_profiles(db: Database) -> list[dict[str, Any]]:
    """Load every stored role profile from the database.

    Queries ``specialty_profiles`` directly via raw SQL because the
    ``Database`` public API only exposes per-specialty lookups.

    Returns:
        List of dicts with keys: ``specialty``, ``role_name``,
        ``defining_skills`` (list of [lemma, weight]), and
        ``supporting_skills`` (list of [lemma, weight]).
    """
    raw = db.connection.execute(
        """
        SELECT specialty, role_name, defining_skills, supporting_skills
        FROM specialty_profiles
        ORDER BY specialty, role_name
        """
    ).fetchall()

    profiles: list[dict[str, Any]] = []
    for speciality, role_name, def_raw, sup_raw in raw:
        # JSON columns come back as Python strings from DuckDB.
        defining: list = json.loads(def_raw) if isinstance(def_raw, str) else (def_raw or [])
        supporting: list = json.loads(sup_raw) if isinstance(sup_raw, str) else (sup_raw or [])
        profiles.append({
            "specialty": speciality,
            "role_name": role_name,
            "defining_skills": defining,
            "supporting_skills": supporting,
        })
    return profiles


def _role_skill_set(role: dict[str, Any]) -> set[str]:
    """Extract the union of a role's defining + supporting skill lemmas."""
    lemmas: set[str] = set()
    for skill_list in (role["defining_skills"], role["supporting_skills"]):
        for entry in skill_list:
            lemma = (entry[0] if isinstance(entry, list) else str(entry)).lower()
            lemmas.add(lemma)
    return lemmas


# ------------------------------------------------------------------
# Jaccard helpers
# ------------------------------------------------------------------


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets.

    Returns 0.0 when union is empty.
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _jaccard_vectorized(
    grad_skills: set[str],
    role_skill_sets: list[set[str]],
) -> np.ndarray:
    """Compute Jaccard similarity between one graduate and many roles.

    Uses set operations for clarity at the scale of role profiles
    (typically < 100 roles across all specialties).
    """
    scores = np.empty(len(role_skill_sets), dtype=np.float64)
    for i, role_set in enumerate(role_skill_sets):
        scores[i] = _jaccard(grad_skills, role_set)
    return scores


# ------------------------------------------------------------------
# Main simulation function
# ------------------------------------------------------------------


def simulate_graduate_match(
    db: Database,
    pool: GraduatePool,
    *,
    top_n: int = 5,
    match_threshold: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Simulate graduate matching against all market roles.

    For each graduate in the pool, computes Jaccard similarity between
    the graduate's skill set and every role profile's combined
    (defining + supporting) skills.  From this produces a match report
    with per-graduate, per-specialty, and overall statistics.

    Args:
        db: ``Database`` with role profiles populated (via
            ``SkillClusterer.discover_roles()``).
        pool: ``GraduatePool`` of graduate curricula to evaluate.
        top_n: Number of top-matching roles to report per graduate.
        match_threshold: Minimum Jaccard score to count a role as
            "qualified" (used for market coverage calculation).
        verbose: Print a rich-formatted report to the console.

    Returns:
        Dict with three sections:

        **per_graduate** (``list[dict]``):
            Per-graduate entries with ``matched_roles``, ``skill_gaps``,
            ``unused_skills``, ``market_coverage_pct``, and
            ``best_match_score``.

        **per_specialty** (``dict[str, dict]``):
            Per-specialty summaries: ``graduate_count``,
            ``avg_match_score``, ``best_role``, ``worst_gap_skill``.

        **overall** (``dict``):
            ``overall_market_coverage``, ``graduates_with_no_match``,
            ``total_graduates``, ``total_roles``.
    """
    # ---- Load roles ----
    roles = _load_all_role_profiles(db)
    if not roles:
        result = {
            "per_graduate": [],
            "per_specialty": {},
            "overall": {
                "overall_market_coverage": 0.0,
                "graduates_with_no_match": 0,
                "total_graduates": 0,
                "total_roles": 0,
            },
        }
        if verbose:
            console = Console()
            console.print(
                "[yellow]No role profiles found in database. "
                "Run 'discover-roles' first.[/yellow]"
            )
        return result

    role_meta: list[dict[str, str]] = [
        {"specialty": r["specialty"], "role_name": r["role_name"]} for r in roles
    ]
    role_skill_sets: list[set[str]] = [_role_skill_set(r) for r in roles]

    # Build global market skill universe for unused-skill detection.
    all_market_skills: set[str] = set()
    for r_set in role_skill_sets:
        all_market_skills |= r_set

    total_roles = len(roles)
    per_graduate: list[dict[str, Any]] = []
    specialty_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for grad in pool:
        grad_set = grad.skill_set

        # Jaccard scores for every role.
        scores = _jaccard_vectorized(grad_set, role_skill_sets)

        # Top-N roles.
        top_indices = np.argsort(scores)[::-1][:top_n]
        matched_roles: list[dict[str, Any]] = []
        skill_gaps: dict[str, list[str]] = {}
        for idx in top_indices:
            score = float(scores[idx])
            role = role_meta[idx]
            r_set = role_skill_sets[idx]
            gaps = sorted(r_set - grad_set)
            matched_roles.append({
                "role_name": role["role_name"],
                "specialty": role["specialty"],
                "score": round(score, 4),
            })
            skill_gaps[role["role_name"]] = gaps

        # Unused skills — graduate has them, no role requires them.
        unused = sorted(grad_set - all_market_skills)

        # Market coverage: fraction of roles with Jaccard ≥ threshold.
        qualifying = int(np.sum(scores >= match_threshold))
        coverage_pct = round(qualifying / total_roles * 100.0, 1)

        best_score = float(np.max(scores)) if total_roles > 0 else 0.0

        entry = {
            "specialty_name": grad.specialty_name,
            "source": grad.source,
            "skills_count": len(grad.skills),
            "matched_roles": matched_roles,
            "skill_gaps": skill_gaps,
            "unused_skills": unused,
            "market_coverage_pct": coverage_pct,
            "best_match_score": round(best_score, 4),
        }
        per_graduate.append(entry)
        specialty_groups[grad.specialty_name].append(entry)

    # ---- Per-specialty aggregates ----
    per_specialty: dict[str, dict[str, Any]] = {}
    for spec_name, entries in specialty_groups.items():
        avg_score = (
            sum(e["best_match_score"] for e in entries) / len(entries)
            if entries
            else 0.0
        )

        # Best role: the one that appears most often as top match.
        top_role_counter: Counter[str] = Counter()
        for e in entries:
            if e["matched_roles"]:
                top_role_counter[e["matched_roles"][0]["role_name"]] += 1

        # Worst gap skill: skill that most graduates are missing.
        gap_counter: Counter[str] = Counter()
        for e in entries:
            for gaps in e["skill_gaps"].values():
                gap_counter.update(gaps)

        per_specialty[spec_name] = {
            "graduate_count": len(entries),
            "avg_match_score": round(avg_score, 4),
            "best_role": top_role_counter.most_common(1)[0][0] if top_role_counter else "—",
            "worst_gap_skill": gap_counter.most_common(1)[0][0] if gap_counter else "—",
        }

    # ---- Overall aggregates ----
    no_match = sum(1 for e in per_graduate if not e["matched_roles"])
    overall_coverage = (
        sum(e["market_coverage_pct"] for e in per_graduate) / len(per_graduate)
        if per_graduate
        else 0.0
    )

    result = {
        "per_graduate": per_graduate,
        "per_specialty": per_specialty,
        "overall": {
            "overall_market_coverage": round(overall_coverage, 1),
            "graduates_with_no_match": no_match,
            "total_graduates": len(per_graduate),
            "total_roles": total_roles,
        },
    }

    if verbose:
        _print_report(result)

    return result


# ------------------------------------------------------------------
# Rich output
# ------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    """Render the match report as rich tables and panels."""
    console = Console()
    overall = report["overall"]

    # -- Title --
    console.print()
    console.print(
        Panel.fit(
            Text(
                "Graduate-to-Market Match Simulation",
                style="bold white",
            ),
            border_style="blue",
        )
    )

    # -- Overall summary --
    console.print(
        f"[bold]Total graduates:[/] {overall['total_graduates']}  "
        f"[bold]Total market roles:[/] {overall['total_roles']}  "
        f"[bold]Overall market coverage:[/] [cyan]{overall['overall_market_coverage']}%[/]  "
        f"[bold]No-match graduates:[/] [red]{overall['graduates_with_no_match']}[/]"
    )
    console.print()

    # -- Per-specialty summary table --
    spec_table = Table(title="Per-Specialty Summary", border_style="dim blue")
    spec_table.add_column("Specialty", style="bold cyan")
    spec_table.add_column("Grads", justify="right")
    spec_table.add_column("Avg Score", justify="right")
    spec_table.add_column("Top Role", style="green")
    spec_table.add_column("Top Gap Skill", style="red")

    for spec, stats in sorted(report["per_specialty"].items()):
        spec_table.add_row(
            spec,
            str(stats["graduate_count"]),
            f"{stats['avg_match_score']:.3f}",
            stats["best_role"],
            stats["worst_gap_skill"],
        )
    console.print(spec_table)
    console.print()

    # -- Per-graduate detail table --
    grad_table = Table(
        title="Per-Graduate Match Report (top matching roles)",
        border_style="dim green",
        show_lines=True,
    )
    grad_table.add_column("Graduate", style="bold")
    grad_table.add_column("Skills", justify="right")
    grad_table.add_column("Coverage", justify="right")
    grad_table.add_column("Best Role", style="green")
    grad_table.add_column("Score", justify="right")
    grad_table.add_column("Missing Skills", style="yellow", max_width=40)
    grad_table.add_column("Unused", style="red", max_width=30)

    for entry in report["per_graduate"]:
        top_role = entry["matched_roles"][0] if entry["matched_roles"] else None
        best_role_label = top_role["role_name"] if top_role else "—"
        best_score = f"{top_role['score']:.3f}" if top_role else "—"
        gaps = (
            ", ".join(entry["skill_gaps"].get(best_role_label, [])[:5])
            if top_role and best_role_label != "—"
            else "—"
        )
        unused = ", ".join(entry["unused_skills"][:3]) if entry["unused_skills"] else "—"

        grad_table.add_row(
            f"{entry['specialty_name']}\n[dim]({entry['source']})[/]",
            str(entry["skills_count"]),
            f"{entry['market_coverage_pct']}%",
            best_role_label,
            best_score,
            gaps,
            unused,
        )

    console.print(grad_table)
    console.print()

    # -- Unused skills summary --
    all_unused: Counter[str] = Counter()
    for entry in report["per_graduate"]:
        all_unused.update(entry["unused_skills"])

    if all_unused:
        unused_table = Table(
            title="Most Frequent Unused Skills (grad has them, market doesn't ask)",
            border_style="dim red",
        )
        unused_table.add_column("Skill", style="red")
        unused_table.add_column("Occurrences", justify="right")
        for skill, count in all_unused.most_common(10):
            unused_table.add_row(skill, str(count))
        console.print(unused_table)
        console.print()
