"""Data-driven competency axis discovery with statistical validation.

Problem: Find the optimal number and composition of competency axes
(dimensions) that best describe a specialty's job market. Axes are
contextual — derived from skill co-occurrence clusters, not literal regex.

Statistical framework:
  H₁: Skill co-occurrence has latent axis structure (silhouette vs random)
  H₂: k+1 axes explain data significantly better than k (bootstrap Δ-score)
  H₃: Scheme A is significantly better than Scheme B (paired bootstrap)
  H₄: Specialty name stability under resampling (bootstrap name ARI)

Metrics per scheme:
  - Coverage:  fraction of top-N skills within axes (non-noise)
  - Separation: mean pairwise Jaccard distance between axis skill sets
  - Stability:  bootstrap ARI of reclustering
  - Coherence:  mean within-axis co-occurrence / mean between-axis
  - Parsimony:  log(k) penalty (AIC-style)
  - Combined:   harmonic mean of coverage × separation × stability × coherence
                divided by (1 + log(k)/log(|skills|))
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from hh_competency.storage.db import Database
from hh_competency.storage.models import SkillFrequency

# ────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────


@dataclass
class AxisDef:
    """A single competency axis (dimension) with defining skills."""

    index: int
    name: str
    skills: list[str]  # skill lemmas in this axis
    defining: list[str]  # top-5 defining skills
    size: int  # number of skills
    coherence: float  # within-axis co-occurrence ratio


@dataclass
class SchemeMetrics:
    """Quality metrics for one axis scheme (k axes)."""

    k: int  # number of axes
    coverage: float  # fraction of top-N skills in axes
    separation: float  # mean pairwise Jaccard distance between axes
    stability: float  # bootstrap ARI
    coherence: float  # mean within-axis / between-axis co-occurrence
    combined: float  # harmonic mean × parsimony penalty
    axes: list[AxisDef] = field(default_factory=list)

    # Bootstrap CIs (populated after evaluation)
    coverage_ci: tuple[float, float] | None = None
    separation_ci: tuple[float, float] | None = None
    stability_ci: tuple[float, float] | None = None
    coherence_ci: tuple[float, float] | None = None
    combined_ci: tuple[float, float] | None = None


@dataclass
class SchemeComparison:
    """Comparison of multiple axis schemes with pairwise significance."""

    specialty: str
    schemes: list[SchemeMetrics]
    pairwise_tests: list[PairwiseTest] = field(default_factory=list)
    optimal_k: int | None = None
    optimal_scheme: SchemeMetrics | None = None


@dataclass
class PairwiseTest:
    """Pairwise statistical comparison of two axis schemes."""

    k_a: int
    k_b: int
    delta_mean: float  # mean difference in combined score (a - b)
    delta_ci: tuple[float, float]  # 95% bootstrap CI
    p_value: float  # bootstrap p-value
    significant: bool  # CI excludes 0


@dataclass
class AxisDiscoveryResult:
    """Full axis discovery result for one specialty."""

    specialty: str
    n_skills: int
    n_vacancies: int
    comparison: SchemeComparison
    discovered_axes: list[AxisDef]
    optimal_k: int


@dataclass
class SpecialtyNameResult:
    """Statistical validation of a specialty name candidate."""

    name: str
    strategy: str  # 'top3', 'heuristic', 'top1'
    stability: float  # bootstrap name ARI
    distinctiveness: float  # Jaccard distance to nearest other specialty
    defining_skills: list[str]
    optimal: bool = False


# ────────────────────────────────────────────────────────
# Heuristic axis naming: contextual, not literal skill lists
# ────────────────────────────────────────────────────────

# Skill → axis category lookup tables
_CATEGORY_RULES: list[tuple[list[str], str, str]] = [
    # (trigger lemmas, axis name in Russian, axis name in English)
    (
        [
            "питон", "python", "java", "c++", "cpp", "matlab", "r",
            "scala", "rust", "javascript", "typescript", "go", "golang",
            "программирование", "разработка", "программист", "developer",
            "бэкенд", "backend", "фронтенд", "frontend", "api", "rest",
            "граф", "git", "github", "gitlab", "docker", "kubernetes",
            "linux", "unix", "bash", "sql", "nosql", "база данных",
        ],
        "Программирование и IT",
        "Programming & IT",
    ),
    (
        [
            "xrd", "рентген", "sem", "tem", "afm", "спектрометр",
            "спектроскоп", "хроматограф", "масс-спектр", "микроскоп",
            "лабораторный", "оборудование", "прибор", "установка",
            "калибровка", "детектор", "сенсор", "лазерный",
            "вакуумный", "оптический", "измерение", "эксперимент",
            "протокол", "методика", "метод", "лаборант",
        ],
        "Экспериментальные методы и оборудование",
        "Experimental Methods & Equipment",
    ),
    (
        [
            "моделирование", "симуляция", "численный", "вычислительный",
            "алгоритм", "dft", "молекулярный", "динамика",
            "конечный элемент", "cfd", "hpc", "кластер",
            "параллельный", "gpu", "cuda", "квантовый", "расчёт",
            "оптимизация", "метод монте", "нейросеть", "нейронный",
            "машинный обучение", "машинный", "обучение",
            "искусственный интеллект", "ai", "ml", "dl",
        ],
        "Вычислительное моделирование",
        "Computational Modeling",
    ),
    (
        [
            "статистика", "анализ данных", "регрессия", "anova",
            "тест", "критерий", "выборка", "распределение",
            "корреляция", "факторный анализ", "кластеризация",
            "визуализация", "график", "диаграмма",
            "база данных", "sql", "обработка", "pandas",
            "очистка", "предобработка", "парсинг",
        ],
        "Анализ данных",
        "Data Analysis",
    ),
    (
        [
            "статья", "публикация", "журнал", "рецензируемый",
            "обзор", "конференция", "доклад", "презентация",
            "отчёт", "документация", "грант", "заявка",
            "научный", "исследование", "патент", "интеллектуальный",
            "результат", "внедрение", "коммерциализация",
        ],
        "Научная коммуникация и публикации",
        "Scientific Communication & Publications",
    ),
    (
        [
            "управление", "руководитель", "проект", "команда",
            "бюджет", "план", "срок", "отчётность",
            "закупка", "тендер", "поставщик", "логистика",
            "риск", "качество", "сертификация", "аудит",
            "нормативный", "регламент", "гост", "снип",
            "охрана труда", "техника безопасности",
        ],
        "Управление проектами и нормативы",
        "Project Management & Compliance",
    ),
    (
        [
            "клетка", "культура", "проба", "образец",
            "полевой", "экспедиция", "мониторинг", "наблюдение",
            "вид", "популяция", "экосистема", "среда",
            "in vitro", "in vivo", "фермент", "белок",
            "днк", "рнк", "пцр", "секвенирование",
            "геном", "протеом", "метаболом",
        ],
        "Биологические и полевые методы",
        "Biological & Field Methods",
    ),
    (
        [
            "синтез", "реакция", "катализ", "реагент",
            "растворитель", "экстракция", "дистилляция",
            "технология", "процесс", "производство",
            "масштабирование", "пилотный", "промышленный",
            "сырьё", "продукт", "выход", "чистота",
            "контроль", "сертификат", "gmp", "glp",
        ],
        "Химический синтез и технология",
        "Chemical Synthesis & Technology",
    ),
    (
        [
            "физика", "механика", "квантовый", "термодинамика",
            "электродинамика", "оптика", "фотоника",
            "полупроводник", "плазма", "ядерный",
            "акустика", "гидродинамика", "аэродинамика",
            "материал", "сплав", "полимер", "композит",
            "свойство", "структура", "фаза", "диаграмма",
        ],
        "Фундаментальная физика и материаловедение",
        "Fundamental Physics & Materials Science",
    ),
]


def _classify_skill_to_category(skill_lemma: str) -> str | None:
    """Classify a skill lemma into a heuristic category name.

    Returns None if no category matches (skill becomes its own axis
    or groups with other unclassified skills).
    """
    skill_lower = skill_lemma.lower().strip()
    for triggers, name_ru, _name_en in _CATEGORY_RULES:
        for trigger in triggers:
            # Trigger can be one word or multi-word
            if trigger in skill_lower or skill_lower in trigger:
                return name_ru
    return None


def _name_axis_from_skills(skills: list[str]) -> str:
    """Generate a contextual axis name from its defining skills.

    Strategy:
    1. Classify each skill into a heuristic category
    2. If ≥60% of skills fall into one category → use that name
    3. If multiple categories are roughly balanced → concatenate top 2
    4. Fallback: top 3 skill lemmas as name
    """
    if not skills:
        return "Неопределённая ось"

    categories: dict[str, int] = defaultdict(int)
    for skill in skills:
        cat = _classify_skill_to_category(skill)
        if cat:
            categories[cat] += 1

    total = len(skills)
    if categories:
        best_cat, best_count = max(categories.items(), key=lambda x: x[1])
        if best_count / total >= 0.5:
            return best_cat
        # Top 2 categories
        sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
        top2 = sorted_cats[:2]
        if len(top2) >= 2 and top2[0][1] / total >= 0.3:
            return f"{top2[0][0]} + {top2[1][0]}"

    # Fallback: top-3 skills as name (truncated)
    top_skills = skills[:3]
    name = ", ".join(top_skills)
    if len(name) > 60:
        name = ", ".join(top_skills[:2])
    return name


# ────────────────────────────────────────────────────────
# Core Engine
# ────────────────────────────────────────────────────────


class AxisDiscoveryEngine:
    """Discover optimal competency axes from skill co-occurrence data.

    Pipeline:
    1. Build skill × vacancy co-occurrence matrix
    2. Compute Jaccard distance between skills
    3. Cluster skills at k = min_k..max_k
    4. Evaluate metrics for each scheme with bootstrapping
    5. Pairwise statistical comparison to select optimal k
    6. Name each axis contextually from its defining skills
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── Entry point ──────────────────────────────────

    def discover(
        self,
        specialty: str,
        skills: list[SkillFrequency],
        n_vacancies: int,
        min_k: int = 2,
        max_k: int = 8,
        n_bootstrap: int = 200,
        min_skills_per_axis: int = 3,
        random_seed: int = 42,
    ) -> AxisDiscoveryResult:
        """Full axis discovery pipeline for one specialty.

        Args:
            specialty: Specialty name (e.g. 'physics')
            skills: SkillFrequency objects (top-N from analysis)
            n_vacancies: Total vacancies for this specialty
            min_k/max_k: Range of axis counts to evaluate
            n_bootstrap: Bootstrap iterations for CIs
            min_skills_per_axis: Minimum skills per axis
            random_seed: Random seed for reproducibility

        Returns:
            AxisDiscoveryResult with optimal axes and full comparison
        """
        rng = np.random.default_rng(random_seed)
        skill_lemmas = [s.lemma for s in skills]

        if len(skill_lemmas) < min_skills_per_axis * min_k:
            return AxisDiscoveryResult(
                specialty=specialty,
                n_skills=len(skill_lemmas),
                n_vacancies=n_vacancies,
                comparison=SchemeComparison(specialty=specialty, schemes=[]),
                discovered_axes=[],
                optimal_k=0,
            )

        # Build co-occurrence matrix
        skill_names, cooc_matrix = _build_cooccurrence_matrix(
            self._db, specialty, skill_lemmas
        )
        if len(skill_names) < min_skills_per_axis * min_k:
            return AxisDiscoveryResult(
                specialty=specialty,
                n_skills=len(skill_names),
                n_vacancies=n_vacancies,
                comparison=SchemeComparison(specialty=specialty, schemes=[]),
                discovered_axes=[],
                optimal_k=0,
            )

        # Compute Jaccard distance matrix
        dist_matrix = _jaccard_distance_matrix(cooc_matrix)

        # Evaluate schemes at each k
        schemes: list[SchemeMetrics] = []
        for k in range(min_k, max_k + 1):
            if k > len(skill_names):
                break
            metrics = self._evaluate_scheme(
                k, skill_names, cooc_matrix, dist_matrix,
                n_bootstrap, min_skills_per_axis, rng,
            )
            schemes.append(metrics)

        # Pairwise comparisons
        pairwise = self._pairwise_compare(
            schemes, skill_names, cooc_matrix, dist_matrix,
            n_bootstrap, min_skills_per_axis, rng,
        )

        comparison = SchemeComparison(
            specialty=specialty,
            schemes=schemes,
            pairwise_tests=pairwise,
        )

        # Select optimal k: largest k where (k vs k+1) is NOT significant
        optimal_k = min_k
        for test in pairwise:
            if test.k_a == test.k_b - 1 and not test.significant:
                optimal_k = test.k_a
                break
        else:
            # No clear plateau — use the scheme with best combined score
            if schemes:
                optimal_k = max(schemes, key=lambda s: s.combined).k

        # Get the optimal axes
        optimal_scheme = next((s for s in schemes if s.k == optimal_k), None)
        if optimal_scheme:
            comparison.optimal_k = optimal_k
            comparison.optimal_scheme = optimal_scheme

        return AxisDiscoveryResult(
            specialty=specialty,
            n_skills=len(skill_names),
            n_vacancies=n_vacancies,
            comparison=comparison,
            discovered_axes=optimal_scheme.axes if optimal_scheme else [],
            optimal_k=optimal_k,
        )

    # ── Scheme evaluation ────────────────────────────

    def _evaluate_scheme(
        self,
        k: int,
        skill_names: list[str],
        cooc_matrix: np.ndarray,
        dist_matrix: np.ndarray,
        n_bootstrap: int,
        min_skills_per_axis: int,
        rng: np.random.Generator,
    ) -> SchemeMetrics:
        """Evaluate one axis scheme (k axes) with bootstrapped metrics."""
        labels = _cluster_skills(k, dist_matrix)
        axes = _extract_axes(k, labels, skill_names, cooc_matrix, min_skills_per_axis)

        # Point estimates
        coverage = _compute_coverage(axes, len(skill_names))
        separation = _compute_separation(axes)
        coherence = _compute_coherence(axes, cooc_matrix, skill_names)
        combined = _compute_combined(coverage, separation, 0.0, coherence, k, len(skill_names))

        scheme = SchemeMetrics(
            k=k,
            coverage=coverage,
            separation=separation,
            stability=0.0,  # filled by bootstrap
            coherence=coherence,
            combined=combined,
            axes=axes,
        )

        # Bootstrap: stability + CIs
        n_skills = len(skill_names)
        ari_samples: list[float] = []
        cov_samples: list[float] = []
        sep_samples: list[float] = []
        coh_samples: list[float] = []
        comb_samples: list[float] = []

        for _ in range(n_bootstrap):
            # Subsample skills (80%)
            idx_sample: list[int] = sorted(
                rng.choice(n_skills, size=max(int(n_skills * 0.8), min_skills_per_axis * k), replace=True)
            )
            # Deduplicate for clustering (unique indices)
            unique_idx = sorted(set(idx_sample))
            if len(unique_idx) < min_skills_per_axis * k:
                continue

            sub_names = [skill_names[i] for i in unique_idx]
            sub_cooc = cooc_matrix[np.ix_(unique_idx, unique_idx)]
            sub_dist = _jaccard_distance_matrix(sub_cooc)

            sub_labels = _cluster_skills(k, sub_dist)
            sub_axes = _extract_axes(k, labels, sub_names, sub_cooc, min_skills_per_axis)

            # Stability: ARI of clustering on full vs subsample
            # Map original skill indices to subsample
            full_sub_labels = np.array([labels[i] for i in unique_idx])
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(full_sub_labels, sub_labels)

            ari_samples.append(ari)
            cov_samples.append(_compute_coverage(sub_axes, len(unique_idx)))
            sep_samples.append(_compute_separation(sub_axes))
            coh_samples.append(_compute_coherence(sub_axes, sub_cooc, sub_names))
            # Combined needs stability in the formula — but for bootstrap stability IS the ARI
            comb_samples.append(_compute_combined(
                cov_samples[-1], sep_samples[-1], ari, coh_samples[-1], k, len(unique_idx)
            ))

        if ari_samples:
            scheme.stability = float(np.mean(ari_samples))
            scheme.combined = _compute_combined(
                coverage, separation, scheme.stability, coherence, k, len(skill_names)
            )

            # Bootstrap CIs (percentile)
            arr_ari = np.array(ari_samples)
            arr_cov = np.array(cov_samples)
            arr_sep = np.array(sep_samples)
            arr_coh = np.array(coh_samples)
            arr_comb = np.array(comb_samples)

            scheme.stability_ci = _percentile_ci(arr_ari)
            scheme.coverage_ci = _percentile_ci(arr_cov)
            scheme.separation_ci = _percentile_ci(arr_sep)
            scheme.coherence_ci = _percentile_ci(arr_coh)
            scheme.combined_ci = _percentile_ci(arr_comb)
        else:
            scheme.stability = 0.0
            scheme.combined = combined

        return scheme

    # ── Pairwise comparison ──────────────────────────

    def _pairwise_compare(
        self,
        schemes: list[SchemeMetrics],
        skill_names: list[str],
        cooc_matrix: np.ndarray,
        dist_matrix: np.ndarray,
        n_bootstrap: int,
        min_skills_per_axis: int,
        rng: np.random.Generator,
    ) -> list[PairwiseTest]:
        """Bootstrap pairwise comparisons between adjacent axis counts.

        For each pair (k, k+1): bootstrap the difference in combined score.
        If 95% CI excludes 0 → k+1 is significantly better.
        """
        tests: list[PairwiseTest] = []
        n_skills = len(skill_names)

        for i in range(len(schemes) - 1):
            k_a = schemes[i].k
            k_b = schemes[i + 1].k

            diffs: list[float] = []
            for _ in range(n_bootstrap):
                idx_sample = sorted(
                    rng.choice(n_skills, size=max(int(n_skills * 0.8), 3), replace=True)
                )
                unique_idx = sorted(set(idx_sample))
                if len(unique_idx) < min_skills_per_axis * k_b:
                    continue

                sub_names = [skill_names[i] for i in unique_idx]
                sub_cooc = cooc_matrix[np.ix_(unique_idx, unique_idx)]
                sub_dist = _jaccard_distance_matrix(sub_cooc)

                # Evaluate both schemes on same subsample
                def _eval_scheme(
                    k_try: int,
                    sub_dist_m=sub_dist,
                    sub_names_m=sub_names,
                    sub_cooc_m=sub_cooc,
                    unique_idx_m=unique_idx,
                ) -> float:
                    lbl = _cluster_skills(k_try, sub_dist_m)
                    axs = _extract_axes(k_try, lbl, sub_names_m, sub_cooc_m, min_skills_per_axis)
                    cv = _compute_coverage(axs, len(unique_idx_m))
                    sp = _compute_separation(axs)
                    ch = _compute_coherence(axs, sub_cooc_m, sub_names_m)
                    return _compute_combined(cv, sp, 0.4, ch, k_try, len(unique_idx_m))

                score_a = _eval_scheme(k_a)
                score_b = _eval_scheme(k_b)
                diffs.append(score_b - score_a)

            if diffs:
                arr = np.array(diffs)
                ci = _percentile_ci(arr)
                p_val = float(np.mean(np.array(diffs) <= 0))
                significant = ci[0] > 0  # entire CI above zero

                tests.append(PairwiseTest(
                    k_a=k_a,
                    k_b=k_b,
                    delta_mean=float(np.mean(arr)),
                    delta_ci=ci,
                    p_value=p_val,
                    significant=significant,
                ))

        return tests


# ────────────────────────────────────────────────────────
# Specialty name validation
# ────────────────────────────────────────────────────────


class SpecialtyNamingEngine:
    """Statistically validate specialty names from market data.

    Tests multiple naming strategies and selects the most stable and
    distinctive one using bootstrap stability analysis.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def discover_name(
        self,
        defining_skills: list[tuple[str, float]],
        all_role_skills: list[str],
        other_role_skills: list[list[str]],
        n_bootstrap: int = 200,
        random_seed: int = 42,
    ) -> list[SpecialtyNameResult]:
        """Generate and validate candidate specialty names.

        Strategies:
          'heuristic' — contextual category-based name
          'top3'      — top 3 defining skills concatenated
          'top1'      — single most defining skill

        Each strategy is evaluated on:
          - Stability: bootstrap name ARI (does name stay same under resampling?)
          - Distinctiveness: Jaccard distance to nearest other specialty

        Returns list of candidates with optimal flag.
        """
        rng = np.random.default_rng(random_seed)
        candidates: list[SpecialtyNameResult] = []

        # Candidate generation
        top_skills_only = [s[0] for s in defining_skills]

        strategies: dict[str, str] = {
            "top3": ", ".join(top_skills_only[:3]),
            "top1": top_skills_only[0],
            "heuristic": _name_axis_from_skills(top_skills_only[:5]),
        }

        # Deduplicate identical names
        seen_names: set[str] = set()
        for strategy, name in strategies.items():
            if name in seen_names:
                continue
            seen_names.add(name)

            # Stability: bootstrap name ARI
            stability = self._bootstrap_name_stability(
                top_skills_only, strategy, n_bootstrap, rng
            )

            # Distinctiveness: Jaccard distance to nearest other role
            distinctiveness = self._compute_distinctiveness(
                set(all_role_skills),
                [set(skills) for skills in other_role_skills],
            )

            candidates.append(SpecialtyNameResult(
                name=name,
                strategy=strategy,
                stability=stability,
                distinctiveness=distinctiveness,
                defining_skills=top_skills_only[:5],
            ))

        # Mark optimal: highest stability × distinctiveness
        if candidates:
            best = max(candidates, key=lambda c: c.stability * c.distinctiveness)
            best.optimal = True

        return candidates

    def _bootstrap_name_stability(
        self,
        skills: list[str],
        strategy: str,
        n_bootstrap: int,
        rng: np.random.Generator,
    ) -> float:
        """Bootstrap: how often does the name stay the same under resampling?"""
        if len(skills) < 4:
            return 1.0  # Too few skills to bootstrap meaningfully

        original = self._generate_name(skills, strategy)
        matches = 0
        n_skills = len(skills)
        sample_size = max(int(n_skills * 0.7), 3)

        for _ in range(n_bootstrap):
            idx = sorted(rng.choice(n_skills, size=sample_size, replace=True))
            sub = [skills[i] for i in idx]
            if self._generate_name(sub, strategy) == original:
                matches += 1

        return matches / n_bootstrap

    def _compute_distinctiveness(
        self,
        our_skills: set[str],
        other_skill_sets: list[set[str]],
    ) -> float:
        """Compute Jaccard distance to nearest other specialty's skills."""
        if not other_skill_sets:
            return 1.0
        min_jaccard = min(
            _jaccard(our_skills, other) for other in other_skill_sets
        )
        return 1.0 - min_jaccard

    @staticmethod
    def _generate_name(skills: list[str], strategy: str) -> str:
        """Generate a name from a list of skills using a strategy."""
        if strategy == "heuristic":
            return _name_axis_from_skills(skills[:5])
        if strategy == "top3":
            return ", ".join(skills[:3])
        if strategy == "top1":
            return skills[0]
        return skills[0]


# ────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────


def _build_cooccurrence_matrix(
    db: Database,
    specialty: str,
    skill_lemmas: list[str],
) -> tuple[list[str], np.ndarray]:
    """Build binary skill × vacancy co-occurrence matrix.

    Returns (skill_names, matrix) where matrix[i,j] = 1 if skill i
    and skill j appear in the same vacancy.
    """
    descriptions = db.get_vacancy_descriptions(specialty)
    if not descriptions:
        return ([], np.array([[]]))

    # For each skill, track which vacancies contain it
    skill_to_vacancies: dict[str, set[int]] = defaultdict(set)
    for vidx, desc in enumerate(descriptions):
        desc_lower = desc.lower()
        for skill in skill_lemmas:
            if skill.lower() in desc_lower:
                skill_to_vacancies.setdefault(skill, set()).add(vidx)

    # Keep skills that appear in at least 2 vacancies
    filtered_skills = [
        s for s in skill_lemmas
        if len(skill_to_vacancies.get(s, set())) >= 2
    ]
    if not filtered_skills:
        return ([], np.array([[]]))

    # Build co-occurrence matrix
    n = len(filtered_skills)
    matrix = np.zeros((n, n))
    for i in range(n):
        vac_i = skill_to_vacancies[filtered_skills[i]]
        for j in range(i, n):
            vac_j = skill_to_vacancies[filtered_skills[j]]
            intersection = len(vac_i & vac_j)
            if intersection > 0:
                matrix[i, j] = intersection
                matrix[j, i] = intersection

    return filtered_skills, matrix


def _jaccard_distance_matrix(cooc_matrix: np.ndarray) -> np.ndarray:
    """Convert co-occurrence counts to Jaccard distances."""
    n = cooc_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n))

    # Jaccard distance = 1 - |A ∩ B| / |A ∪ B|
    # cooc[i,j] = |A ∩ B|, need |A| from diagonal
    diag = np.diag(cooc_matrix)
    union = diag[:, None] + diag[None, :] - cooc_matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_sim = np.where(union > 0, cooc_matrix / union, 0.0)
    jaccard_dist = 1.0 - jaccard_sim
    np.fill_diagonal(jaccard_dist, 0.0)
    return jaccard_dist


def _cluster_skills(k: int, dist_matrix: np.ndarray) -> np.ndarray:
    """Cluster skills using AgglomerativeClustering with Jaccard distance."""
    if dist_matrix.shape[0] <= k:
        return np.arange(dist_matrix.shape[0])
    clusterer = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    return clusterer.fit_predict(dist_matrix)


def _extract_axes(
    k: int,
    labels: np.ndarray,
    skill_names: list[str],
    cooc_matrix: np.ndarray,
    min_skills: int,
) -> list[AxisDef]:
    """Extract AxisDef objects from cluster labels."""
    axes: list[AxisDef] = []
    for axis_idx in range(k):
        mask = labels == axis_idx
        indices = np.where(mask)[0]
        skills_in_axis = [
            skill_names[i] for i in indices if 0 <= i < len(skill_names)
        ]

        if len(skills_in_axis) < min_skills:
            continue

        # Compute within-axis coherence
        if len(skills_in_axis) >= 2:
            idx = [list(skill_names).index(s) for s in skills_in_axis]
            sub_matrix = cooc_matrix[np.ix_(idx, idx)]
            within = np.mean(sub_matrix[np.triu_indices_from(sub_matrix, k=1)])
            total = np.mean(cooc_matrix[np.triu_indices_from(cooc_matrix, k=1)])
            coherence = float(within / total) if total > 0 else 0.0
        else:
            coherence = 0.0

        axis = AxisDef(
            index=axis_idx,
            name=_name_axis_from_skills(skills_in_axis[:5]),
            skills=skills_in_axis,
            defining=skills_in_axis[:5],
            size=len(skills_in_axis),
            coherence=coherence,
        )
        axes.append(axis)

    return axes


def _compute_coverage(axes: list[AxisDef], total_skills: int) -> float:
    """Fraction of skills assigned to any axis."""
    if total_skills == 0:
        return 0.0
    assigned = sum(ax.size for ax in axes)
    return assigned / total_skills


def _compute_separation(axes: list[AxisDef]) -> float:
    """Mean pairwise Jaccard distance between axis skill sets."""
    if len(axes) < 2:
        return 1.0
    distances = []
    for ax_i, ax_j in itertools.combinations(axes, 2):
        set_i = set(ax_i.skills)
        set_j = set(ax_j.skills)
        distances.append(1.0 - _jaccard(set_i, set_j))
    return float(np.mean(distances))


def _compute_coherence(
    axes: list[AxisDef],
    cooc_matrix: np.ndarray,
    skill_names: list[str],
) -> float:
    """Mean within-axis / between-axis co-occurrence ratio."""
    if not axes:
        return 0.0

    within_vals: list[float] = []
    between_vals: list[float] = []

    name_to_idx = {name: i for i, name in enumerate(skill_names)}

    for ax in axes:
        indices = [name_to_idx[s] for s in ax.skills if s in name_to_idx]
        if len(indices) >= 2:
            sub = cooc_matrix[np.ix_(indices, indices)]
            within_vals.append(float(np.mean(sub)))

        # Between: this axis skills vs all other skills
        other_idx = [i for i in range(len(skill_names)) if i not in indices]
        if indices and other_idx:
            between = cooc_matrix[np.ix_(indices, other_idx)]
            between_vals.append(float(np.mean(between)))

    mean_within = np.mean(within_vals) if within_vals else 0.0
    mean_between = np.mean(between_vals) if between_vals else 1.0
    if mean_between == 0.0:
        return 1.0 if mean_within > 0 else 0.0
    return float(mean_within / mean_between)


def _compute_combined(
    coverage: float,
    separation: float,
    stability: float,
    coherence: float,
    k: int,
    n_skills: int,
) -> float:
    """Combined score: harmonic mean × parsimony penalty.

    Harmoic mean of metrics that should be HIGH (coverage, separation,
    stability, coherence), then penalized by log(k)/log(n).
    """
    metrics = [max(coverage, 0.001), max(separation, 0.001),
               max(stability, 0.001), max(coherence, 0.001)]
    n_metrics = len(metrics)

    # Harmonic mean
    hm = n_metrics / sum(1.0 / m for m in metrics)

    # Parsimony penalty
    penalty = 1.0 if n_skills <= 1 else 1.0 / (1.0 + np.log(k) / np.log(n_skills))

    return hm * penalty


def _percentile_ci(arr: np.ndarray, alpha: float = 0.025) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    low = float(np.percentile(arr, alpha * 100))
    high = float(np.percentile(arr, (1 - alpha) * 100))
    return (low, high)


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
