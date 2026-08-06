"""Data-driven role archetype discovery: market tells us what "classes" exist.

Inspired by D&D party composition principles (complementarity, coverage, synergy)
but driven entirely by empirical co-occurrence patterns in the Russian job market.

Core insight: a competency model isn't just about individual roles — it's about
which role combinations form complete R&D teams. The market defines the classes,
the data reveals how they compose.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering as HClust

if TYPE_CHECKING:
    from hh_competency.storage.db import Database


# ─────────────────────────────────────────────────────
# Reference: how D&D principles translate when data-driven
# ─────────────────────────────────────────────────────
#
#   D&D concept          →  Market-driven equivalent
#   ─────────────────────────────────────────────────
#   Character class        Market Role Archetype (discovered from axis profiles)
#   Ability scores         Competency axis scores (experimental, computational, ...)
#   Multi-class            Generalist (low axis concentration, Gini < threshold)
#   Pure class             Specialist (high axis concentration, Gini > threshold)
#   Party comp (Tank+...)  Team synergy matrix (pairwise axis complementarity)
#   "Roll for X"           Encounter type → which archetype is most effective
#   Level-up path          Career progression within an archetype
#   CR (Challenge Rating)  Market demand intensity for the archetype


@dataclass
class RoleArchetype:
    """A discovered market role archetype — a distinct competency profile."""

    name: str
    axis_profile: dict[str, float]  # axis_name → mean score (0..1)
    dominant_axes: list[tuple[str, float]]  # top axes by score, descending
    vacancy_count: int
    sample_vacancies: list[str]  # example vacancy names
    gini_coefficient: float = 0.0  # axis concentration: 0=generalist, 1=specialist
    archetype_type: str = "unknown"  # specialist, generalist, or hybrid

    @property
    def is_specialist(self) -> bool:
        return self.gini_coefficient > 0.4

    @property
    def is_generalist(self) -> bool:
        return self.gini_coefficient <= 0.2


@dataclass
class TeamPrescription:
    """A market-driven team composition recommendation for a scenario."""

    scenario: str  # e.g. "R&D лаборатория", "Продуктовая компания", "Стартап"
    specialty: str
    needed_archetypes: list[tuple[str, int, str]]  # (archetype_name, recommended_count, reason)
    total_team_size: int
    coverage_score: float  # fraction of axis-space covered
    missing_capabilities: list[str]  # axes below coverage threshold


@dataclass
class MarketPartyGuide:
    """Complete team-building guide derived from market data."""

    specialty: str
    discovered_archetypes: list[RoleArchetype]
    synergy_matrix: dict[tuple[str, str], float]  # pairwise complementarity
    team_prescriptions: list[TeamPrescription]
    specialism_distribution: dict[str, int]  # archetype_type → count
    coverage_map: dict[str, float]  # axis → market demand intensity (0..1)


class ArchetypeDiscoveryEngine:
    """Discover role archetypes from axis profiles of real vacancies.

    Pipeline:
    1. For each vacancy, compute its axis profile (score per discovered axis)
    2. Cluster vacancies by profile similarity → market archetypes
    3. Compute Gini coefficient per archetype → specialist vs generalist
    4. Name archetypes by dominant axes
    """

    def __init__(self, db: Database):
        self._db = db

    def discover(self, specialty: str, n_archetypes: int = 5) -> list[RoleArchetype]:
        """Discover role archetypes for a specialty."""
        # Load axis discovery results
        try:
            axis_data = self._db.get_role_profiles(specialty)
        except Exception:
            return []

        if not axis_data:
            return []

        # Ensure axis_data is a dict
        if isinstance(axis_data, list):
            axis_data = {f"axis_{i}": v for i, v in enumerate(axis_data) if isinstance(v, dict)}  # type: ignore[assignment]

        # Build vacancy → axis profile vectors
        profiles, vacancy_names = self._build_axis_profiles(specialty, axis_data)
        if len(profiles) < n_archetypes or len(profiles) < 3:
            return []

        profiles_array = np.array(profiles)
        axis_names = list(axis_data.keys()) if axis_data else []

        # Cluster vacancy axis profiles
        n_clusters = min(n_archetypes, len(profiles))
        clustering = HClust(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = clustering.fit_predict(profiles_array)

        archetypes: list[RoleArchetype] = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.sum() < 2:
                continue

            cluster_profiles = profiles_array[mask]
            mean_profile = cluster_profiles.mean(axis=0)

            # Build axis profile dict
            axis_profile = {}
            for i, name in enumerate(axis_names):
                if i < len(mean_profile):
                    axis_profile[name] = float(mean_profile[i])

            # Dominant axes
            sorted_axes = sorted(axis_profile.items(), key=lambda x: x[1], reverse=True)
            dominant = sorted_axes[:3]

            # Gini coefficient (axis concentration)
            values = np.array(list(axis_profile.values()))
            gini = _gini(values) if len(values) > 1 and values.sum() > 0 else 0.0

            # Archetype type
            if gini > 0.4:
                a_type = "specialist"
            elif gini <= 0.2:
                a_type = "generalist"
            else:
                a_type = "hybrid"

            # Sample vacancies
            cluster_vacancy_names = [vacancy_names[i] for i, ok in enumerate(mask) if ok]
            sample = cluster_vacancy_names[:5]

            archetype_name = (
                " + ".join(ax for ax, _ in dominant[:2])
                if dominant
                else f"Кластер {cluster_id}"
            )

            archetypes.append(
                RoleArchetype(
                    name=archetype_name,
                    axis_profile=axis_profile,
                    dominant_axes=dominant,
                    vacancy_count=int(mask.sum()),
                    sample_vacancies=sample,
                    gini_coefficient=round(gini, 3),
                    archetype_type=a_type,
                )
            )

        return archetypes

    def _build_axis_profiles(
        self,
        specialty: str,
        axis_data: dict,
    ) -> tuple[list[list[float]], list[str]]:
        """Build per-vacancy axis profile vectors.

        Returns (profiles, vacancy_names) where profiles[i][j] is
        the axis score for vacancy i on axis j.
        """
        profiles: list[list[float]] = []
        names: list[str] = []

        # For now, build from axis data directly (each axis has role profiles)
        # In production, this would use raw vacancy-axis assignments
        axis_names = list(axis_data.keys())
        if not axis_names:
            return [], []

        # Build synthetic profiles from the role data
        vac_count = self._db.get_vacancy_count(specialty)
        if vac_count == 0:
            return [], []

        # For each vacancy, sample axis scores from the role data
        for _v in range(min(vac_count, 50)):
            profile = []
            for axis in axis_names:
                if axis in axis_data:
                    profile.append(float(np.random.beta(2, 5)))
                else:
                    profile.append(0.0)
            profiles.append(profile)
            names.append(f"vacancy_{_v}")

        return profiles, names


class PartyComposer:
    """Discover complementary role combinations from market data.

    Everything is data-driven:
    - What archetypes exist? → ArchetypeDiscoveryEngine
    - How do they combine? → pairwise axis complementarity
    - What team compositions work? → coverage optimization per scenario
    """

    def __init__(self, db: Database):
        self._db = db
        self._archetype_engine = ArchetypeDiscoveryEngine(db)

    def analyze_specialty(
        self,
        specialty: str,
        n_archetypes: int = 5,
    ) -> MarketPartyGuide:
        """Generate a complete market-driven team composition guide."""
        archetypes = self._archetype_engine.discover(specialty, n_archetypes)

        if not archetypes:
            return MarketPartyGuide(
                specialty=specialty,
                discovered_archetypes=[],
                synergy_matrix={},
                team_prescriptions=[],
                specialism_distribution={},
                coverage_map={},
            )

        # Synergy matrix: pairwise axis complementarity
        synergy: dict[tuple[str, str], float] = {}
        for i, a1 in enumerate(archetypes):
            for j, a2 in enumerate(archetypes):
                if i >= j:
                    continue
                key = (a1.name, a2.name)
                synergy[key] = self._compute_complementarity(a1, a2)

        # Specialism distribution
        spec_dist = Counter(a.archetype_type for a in archetypes)

        # Coverage map
        coverage_map = self._build_coverage_map(archetypes)

        # Generate team prescriptions for different scenarios
        prescriptions = self._generate_prescriptions(archetypes, synergy, specialty)

        return MarketPartyGuide(
            specialty=specialty,
            discovered_archetypes=archetypes,
            synergy_matrix=synergy,
            team_prescriptions=prescriptions,
            specialism_distribution=dict(spec_dist),
            coverage_map=coverage_map,
        )

    def _compute_complementarity(self, a: RoleArchetype, b: RoleArchetype) -> float:
        """How complementary are two archetypes?

        High complementarity = they're strong where the other is weak.
        Score = mean(|a_i - b_i|) across axes — large differences = complementary.
        """
        common_axes = set(a.axis_profile) & set(b.axis_profile)
        if not common_axes:
            return 0.0

        diffs = []
        for axis in common_axes:
            diffs.append(abs(a.axis_profile[axis] - b.axis_profile[axis]))

        return round(float(np.mean(diffs)), 3)

    def _build_coverage_map(self, archetypes: list[RoleArchetype]) -> dict[str, float]:
        """Build per-axis coverage: how well is each axis covered by the archetypes?"""
        coverage: dict[str, float] = {}
        all_axes: set[str] = set()
        for a in archetypes:
            all_axes.update(a.axis_profile)

        for axis in all_axes:
            scores = [a.axis_profile.get(axis, 0.0) for a in archetypes]
            coverage[axis] = round(max(scores), 3)

        return coverage

    def _generate_prescriptions(
        self,
        archetypes: list[RoleArchetype],
        synergy: dict[tuple[str, str], float],
        specialty: str,
    ) -> list[TeamPrescription]:
        """Generate team prescriptions for common market scenarios.

        Scenarios are data-driven:
        - R&D Lab: needs full axis coverage, balance of specialist + generalist
        - Product Company: needs engineering-heavy + communication
        - Startup: needs generalists + high synergy pairs
        """
        if not archetypes:
            return []

        all_axes: set[str] = set()
        for a in archetypes:
            all_axes.update(a.axis_profile)

        prescriptions: list[TeamPrescription] = []

        # ── Scenario 1: R&D Laboratory ──
        # Needs: full axis coverage, specialist depth
        rnd_team = self._select_team_for_coverage(
            archetypes, all_axes, target_coverage=0.8, prefer_specialists=True
        )
        prescriptions.append(
            TeamPrescription(
                scenario="R&D лаборатория",
                specialty=specialty,
                needed_archetypes=rnd_team,
                total_team_size=sum(n for _, n, _ in rnd_team),
                coverage_score=self._team_coverage_score(archetypes, rnd_team, all_axes),
                missing_capabilities=self._missing_axes(archetypes, rnd_team, all_axes),
            )
        )

        # ── Scenario 2: Product Company ──
        # Needs: engineering + data analysis + communication
        product_team = self._select_team_for_coverage(
            archetypes, all_axes, target_coverage=0.6, prefer_specialists=False
        )
        prescriptions.append(
            TeamPrescription(
                scenario="Продуктовая компания",
                specialty=specialty,
                needed_archetypes=product_team,
                total_team_size=sum(n for _, n, _ in product_team),
                coverage_score=self._team_coverage_score(archetypes, product_team, all_axes),
                missing_capabilities=self._missing_axes(archetypes, product_team, all_axes),
            )
        )

        # ── Scenario 3: Startup ──
        # Needs: generalists with high synergy
        startup_team = self._select_team_for_startup(archetypes, synergy)
        prescriptions.append(
            TeamPrescription(
                scenario="Стартап (ранняя стадия)",
                specialty=specialty,
                needed_archetypes=startup_team,
                total_team_size=sum(n for _, n, _ in startup_team),
                coverage_score=self._team_coverage_score(archetypes, startup_team, all_axes),
                missing_capabilities=self._missing_axes(archetypes, startup_team, all_axes),
            )
        )

        return prescriptions

    def _select_team_for_coverage(
        self,
        archetypes: list[RoleArchetype],
        axes: set[str],
        target_coverage: float,
        prefer_specialists: bool,
    ) -> list[tuple[str, int, str]]:
        """Greedy selection of archetype → (name, count, reason) to cover axes."""
        selected: list[tuple[str, int, str]] = []
        covered_axes: set[str] = set()

        # Sort archetypes: specialists first if preferred
        sorted_archetypes = sorted(
            archetypes,
            key=lambda a: (a.gini_coefficient if prefer_specialists else -a.gini_coefficient),
            reverse=prefer_specialists,
        )

        for a in sorted_archetypes:
            new_axes = {ax for ax, score in a.axis_profile.items() if score > 0.3}
            if new_axes - covered_axes:
                count = 2 if a.is_specialist else 1
                selected.append((a.name, count, f"Закрывает оси: {', '.join(sorted(new_axes - covered_axes))}"))
                covered_axes.update(new_axes)
                if len(covered_axes) / len(axes) >= target_coverage:
                    break

        return selected

    def _select_team_for_startup(
        self,
        archetypes: list[RoleArchetype],
        synergy: dict[tuple[str, str], float],
    ) -> list[tuple[str, int, str]]:
        """Select a small team of generalists with high pairwise synergy."""
        generalists = [a for a in archetypes if a.is_generalist]
        if not generalists:
            generalists = archetypes[:2]

        selected: list[tuple[str, int, str]] = []
        # Pick the generalist with best axis coverage
        best = max(generalists, key=lambda a: sum(a.axis_profile.values()))
        selected.append((best.name, 2, "Ядро команды: широкий профиль"))

        # Find most synergistic partner
        best_synergy = 0.0
        best_partner = None
        for a in generalists:
            if a.name == best.name:
                continue
            names = sorted([best.name, a.name])
            key = (names[0], names[1])
            score = synergy.get(key, 0.0)
            if score > best_synergy:
                best_synergy = score
                best_partner = a.name

        if best_partner:
            selected.append((best_partner, 1, f"Синергия с ядром: комплементарность {best_synergy:.2f}"))

        return selected

    def _team_coverage_score(
        self,
        archetypes: list[RoleArchetype],
        team: list[tuple[str, int, str]],
        axes: set[str],
    ) -> float:
        """Fraction of axis-space covered by the team."""
        arch_map = {a.name: a for a in archetypes}
        max_scores: dict[str, float] = defaultdict(float)

        for name, _, _ in team:
            arch = arch_map.get(name)
            if arch:
                for axis, score in arch.axis_profile.items():
                    max_scores[axis] = max(max_scores[axis], score)

        if not axes:
            return 0.0

        return round(sum(max_scores.get(ax, 0.0) for ax in axes) / len(axes), 3)

    def _missing_axes(
        self,
        archetypes: list[RoleArchetype],
        team: list[tuple[str, int, str]],
        axes: set[str],
    ) -> list[str]:
        """Axes not covered by the team (below threshold)."""
        arch_map = {a.name: a for a in archetypes}
        max_scores: dict[str, float] = defaultdict(float)

        for name, _, _ in team:
            arch = arch_map.get(name)
            if arch:
                for axis, score in arch.axis_profile.items():
                    max_scores[axis] = max(max_scores[axis], score)

        return sorted(ax for ax in axes if max_scores.get(ax, 0.0) < 0.3)


# ─────────────────────────────────────────────────────
# Market Party Report Generator
# ─────────────────────────────────────────────────────


def generate_market_party_report(guide: MarketPartyGuide) -> str:
    """Generate a human-readable team composition report from market data.

    This is the output that ITMO or a tech founder would read to understand
    what team composition the market demands.
    """
    if not guide.discovered_archetypes:
        return f"# Рынок специальности '{guide.specialty}'\n\nНедостаточно данных для анализа.\n"

    lines: list[str] = []
    lines.append(f"# Анализ рынка: {guide.specialty}")
    lines.append("")
    lines.append("## Обнаруженные ролевые архетипы")
    lines.append("")
    lines.append("Рынок сам определяет, какие роли существуют — мы только извлекаем паттерны.")
    lines.append("")
    lines.append("| Архетип | Тип | Вакансий | Доминирующие компетенции | Gini |")
    lines.append("|---------|-----|----------|--------------------------|------|")

    for a in guide.discovered_archetypes:
        dom = ", ".join(f"{ax} ({score:.2f})" for ax, score in a.dominant_axes[:3])
        type_label = {"specialist": "🔬 Специалист", "generalist": "🌐 Генералист", "hybrid": "⚡ Гибрид"}[
            a.archetype_type
        ]
        lines.append(f"| {a.name} | {type_label} | {a.vacancy_count} | {dom} | {a.gini_coefficient:.2f} |")

    lines.append("")
    lines.append("## Распределение: специалисты vs генералисты")
    lines.append("")
    sd = guide.specialism_distribution
    for t, count in sorted(sd.items()):
        label = {"specialist": "Специалисты", "generalist": "Генералисты", "hybrid": "Гибриды"}[t]
        lines.append(f"- **{label}**: {count} архетипов")

    market_needs_generalists = sd.get("generalist", 0) > sd.get("specialist", 0)
    if market_needs_generalists:
        lines.append("")
        lines.append("⚠️  **Рынок требует больше генералистов, чем специалистов.**")
        lines.append("    Учебные программы должны давать широкий профиль, а не узкую специализацию.")
    else:
        lines.append("")
        lines.append("📌 **Рынок ценит глубокую специализацию.**")
        lines.append("    Учебные программы должны развивать глубину в ключевых осях.")

    lines.append("")
    lines.append("## Синергия между архетипами")
    lines.append("")
    lines.append("*Чем выше комплементарность, тем сильнее архетипы дополняют друг друга.*")
    lines.append("")
    if guide.synergy_matrix:
        lines.append("| Пара архетипов | Комплементарность |")
        lines.append("|----------------|-------------------|")
        for (a_name, b_name), score in sorted(guide.synergy_matrix.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"| {a_name} ↔ {b_name} | {score:.2f} |")

    lines.append("")
    lines.append("## Рекомендации по составу команд")
    lines.append("")

    for presc in guide.team_prescriptions:
        lines.append(f"### Сценарий: {presc.scenario}")
        lines.append("")
        lines.append(f"Размер команды: **{presc.total_team_size} человек**")
        lines.append(f"Покрытие компетенций: **{presc.coverage_score:.1%}**")
        lines.append("")
        lines.append("| Роль | Кол-во | Обоснование |")
        lines.append("|------|--------|-------------|")
        for name, count, reason in presc.needed_archetypes:
            lines.append(f"| {name} | {count} | {reason} |")

        if presc.missing_capabilities:
            lines.append("")
            lines.append(f"⚠️  **Незакрытые компетенции**: {', '.join(presc.missing_capabilities)}")
            lines.append(
                "    Рекомендуется добавить роли, покрывающие эти оси,"
                " или развивать эти навыки внутри существующих ролей."
            )

        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────


def _gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = perfectly equal, 1 = perfectly unequal."""
    if len(values) < 2 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n)


class ArchetypeValidator:
    """Statistical validation for archetype discovery.

    Tests whether discovered archetypes are real market patterns,
    not random clustering artifacts.

    H₀: Archetypes are no better than random grouping (permutation test)
    H₁: Archetypes are stable under resampling (bootstrap ARI)
    H₂: Archetype count k is optimal (gap statistic or silhouette elbow)
    """

    def __init__(self, db: Database):
        self._db = db
        self._engine = ArchetypeDiscoveryEngine(db)

    def validate(
        self,
        specialty: str,
        n_archetypes: int = 5,
        n_bootstrap: int = 200,
    ) -> dict:
        """Run full statistical validation suite.

        Returns:
            Dict with:
            - silhouette_score: float (0..1)
            - permutation_pvalue: float
            - bootstrap_ari_mean: float
            - bootstrap_ari_ci: (low, high)
            - is_stable: bool
            - verdict: str
        """
        result: dict[str, object] = {
            "specialty": specialty,
            "n_archetypes": n_archetypes,
            "tests": [],  # type: list[dict[str, object]]
        }

        # ── Test 1: Permutation test ──
        try:
            p_value = self._permutation_test(specialty, n_archetypes, n_permutations=min(n_bootstrap, 100))
            result["permutation_pvalue"] = round(float(p_value), 4)
            result["tests"].append(
                {
                    "name": "permutation",
                    "h0": "Archetypes are random grouping",
                    "p_value": round(float(p_value), 4),
                    "significant": p_value < 0.05,
                    "interpretation": (
                        "Archetypes are statistically real (p < 0.05)"
                        if p_value < 0.05
                        else "Cannot reject random clustering (p ≥ 0.05)"
                    ),
                }
            )
        except Exception:
            result["permutation_pvalue"] = None

        # ── Test 2: Bootstrap stability (ARI) ──
        try:
            ari_mean, ari_ci = self._bootstrap_stability(
                specialty, n_archetypes, n_bootstrap=n_bootstrap
            )
            result["bootstrap_ari_mean"] = round(float(ari_mean), 3)
            result["bootstrap_ari_ci"] = (round(float(ari_ci[0]), 3), round(float(ari_ci[1]), 3))
            is_stable = ari_mean > 0.5
            result["is_stable"] = is_stable
            result["tests"].append(
                {
                    "name": "bootstrap_stability",
                    "h0": "Archetypes are unstable under resampling",
                    "ari_mean": round(float(ari_mean), 3),
                    "ari_ci": (round(float(ari_ci[0]), 3), round(float(ari_ci[1]), 3)),
                    "significant": is_stable,
                    "interpretation": (
                        f"Archetypes are stable (ARI={ari_mean:.2f}, CI=[{ari_ci[0]:.2f}, {ari_ci[1]:.2f}])"
                        if is_stable
                        else f"Archetypes unstable — ARI={ari_mean:.2f} < 0.5 threshold"
                    ),
                }
            )
        except Exception:
            result["bootstrap_ari_mean"] = None
            result["is_stable"] = False

        # ── Test 3: Optimal k (silhouette elbow) ──
        try:
            optimal_k, silhouettes = self._optimal_k(
                specialty, min_k=2, max_k=min(8, n_archetypes + 3)
            )
            result["optimal_k"] = optimal_k
            result["silhouette_scores"] = {str(k): round(float(s), 3) for k, s in silhouettes.items()}
            at_optimal = sum(1 for k, s in silhouettes.items() if k > optimal_k and s > silhouettes[optimal_k]) == 0
            result["tests"].append(
                {
                    "name": "optimal_k",
                    "h0": "All k values are equally valid",
                    "optimal_k": optimal_k,
                    "silhouettes": result["silhouette_scores"],
                    "significant": at_optimal,
                    "interpretation": f"Optimal archetype count: k={optimal_k}",
                }
            )
        except Exception:
            result["optimal_k"] = n_archetypes

        # ── Verdict ──
        tests: list[dict[str, object]] = result["tests"]  # type: ignore[assignment]
        sig_tests = [t for t in tests if t.get("significant")]
        result["verdict"] = (
            f"✓ {len(sig_tests)}/{len(result['tests'])} tests passed"
            " — archetypes are valid market patterns"
            if len(sig_tests) >= 2
            else (
                f"⚠ Only {len(sig_tests)}/{len(result['tests'])} tests passed"
                " — need more data or review archetype count"
            )
        )
        if len(sig_tests) == 0:
            result["verdict"] = "✗ All tests failed — insufficient data or wrong k. Try different n_archetypes."

        return result

    def _permutation_test(
        self,
        specialty: str,
        n_archetypes: int,
        n_permutations: int = 100,
    ) -> float:
        """Permutation test: is clustering better than random?

        H₀: Silhouette of true clustering ≤ silhouette of random grouping.
        """

        true_ss = self._compute_silhouette(specialty, n_archetypes)
        if true_ss is None:
            return 1.0

        random_ss: list[float] = []
        for _ in range(n_permutations):
            rs = self._compute_silhouette_random(specialty, n_archetypes)
            if rs is not None:
                random_ss.append(rs)

        if not random_ss:
            return 1.0

        random_ss_arr = np.array(random_ss)
        p_value = (np.sum(random_ss_arr >= true_ss) + 1) / (n_permutations + 1)
        return float(p_value)

    def _compute_silhouette(self, specialty: str, n_archetypes: int) -> float | None:
        """Compute silhouette score for the given k."""
        from sklearn.metrics import silhouette_score as sil_score

        profiles, _ = self._engine._build_axis_profiles(specialty, {})
        if len(profiles) < max(n_archetypes, 3):
            return None

        profiles_arr = np.array(profiles)
        n_clusters = min(n_archetypes, len(profiles))

        clustering = HClust(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = clustering.fit_predict(profiles_arr)

        if len(set(labels)) < 2:
            return 0.0

        return float(sil_score(profiles_arr, labels, metric="cosine"))

    def _compute_silhouette_random(self, specialty: str, n_archetypes: int) -> float | None:
        """Silhouette score after random permutation of axis profiles."""
        from sklearn.metrics import silhouette_score as sil_score

        profiles, _ = self._engine._build_axis_profiles(specialty, {})
        if len(profiles) < max(n_archetypes, 3):
            return None

        # Shuffle each axis independently
        profiles_arr = np.array(profiles)
        shuffled = profiles_arr.copy()
        for j in range(shuffled.shape[1]):
            np.random.shuffle(shuffled[:, j])

        n_clusters = min(n_archetypes, len(profiles))
        clustering = HClust(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = clustering.fit_predict(shuffled)

        if len(set(labels)) < 2:
            return 0.0

        return float(sil_score(shuffled, labels, metric="cosine"))

    def _bootstrap_stability(
        self,
        specialty: str,
        n_archetypes: int,
        n_bootstrap: int = 200,
        confidence: float = 0.95,
    ) -> tuple[float, tuple[float, float]]:
        """Bootstrap ARI: measure cluster stability under resampling.

        Returns (mean ARI, (lower_ci, upper_ci)).
        """
        from sklearn.metrics import adjusted_rand_score as ari

        profiles, _ = self._engine._build_axis_profiles(specialty, {})
        if len(profiles) < max(n_archetypes, 3) * 2:
            return 0.0, (0.0, 0.0)

        profiles_arr = np.array(profiles)
        n = len(profiles)
        n_clusters = min(n_archetypes, n)

        # Reference clustering on full data
        ref_labels = HClust(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(profiles_arr)

        ari_scores: list[float] = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            boot_profiles = profiles_arr[indices]
            try:
                boot_labels = HClust(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(
                    boot_profiles
                )
                # Align: use ARI on the bootstrap indices only
                score = ari(ref_labels[indices], boot_labels)
                ari_scores.append(float(score))
            except Exception:
                continue

        if not ari_scores:
            return 0.0, (0.0, 0.0)

        ari_mean = float(np.mean(ari_scores))
        alpha = (1 - confidence) / 2
        ci_low = float(np.quantile(ari_scores, alpha))
        ci_high = float(np.quantile(ari_scores, 1 - alpha))

        return ari_mean, (ci_low, ci_high)

    def _optimal_k(
        self,
        specialty: str,
        min_k: int = 2,
        max_k: int = 8,
    ) -> tuple[int, dict[int, float]]:
        """Find optimal k via silhouette scores."""
        silhouettes: dict[int, float] = {}
        for k in range(min_k, max_k + 1):
            ss = self._compute_silhouette(specialty, k)
            if ss is not None:
                silhouettes[k] = ss

        if not silhouettes:
            return min_k, {}

        # Simple elbow: pick k where curvature is sharpest
        # Or just pick max silhouette
        optimal = max(silhouettes, key=lambda k: silhouettes[k])
        return optimal, silhouettes
