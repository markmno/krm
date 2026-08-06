"""Curriculum recommendation engine: skill tokens → course recommendations.

Transforms discovered role profiles and competency axes into concrete
curriculum recommendations with prerequisites, semester planning,
and market-coverage optimization for ITMO university.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from hh_competency.storage.models import SkillFrequency

if TYPE_CHECKING:
    from hh_competency.storage.db import Database

# ───────────────────────────────────────────────────────────────────
# Course catalogue: maps skill patterns to course expressions
# ───────────────────────────────────────────────────────────────────

_COURSE_CATALOGUE: list[tuple[str, str, list[str], list[str]]] = [
    # (course_name, course_category, prerequisite_skills, skill_expressions)
    # A skill_expression is a regex matched against the skill lemma.
    # If ANY match → this course teaches that skill.
    (
        "Программирование на Python",
        "Программирование",
        [],
        ["python", r"программировани", r"скрипт", r"автоматизаци"],
    ),
    (
        "Программирование на C/C++",
        "Программирование",
        ["Программирование на Python"],
        [r"c\+\+", r"\bc\b", r"cpp", r"компиляци"],
    ),
    (
        "Алгоритмы и структуры данных",
        "Программирование",
        ["Программирование на Python"],
        ["алгоритм", "структур данных", "сложность", "сортировк"],
    ),
    (
        "Численные методы",
        "Вычислительные",
        ["Программирование на Python"],
        ["численн", "вычислительн", "математическ"],
    ),
    (
        "Машинное обучение",
        "Анализ данных",
        ["Алгоритмы и структуры данных", "Теория вероятностей и статистика"],
        ["машинное обучение", "нейрон", "глубокое обучение", "scikit", "pytorch"],
    ),
    (
        "Теория вероятностей и статистика",
        "Анализ данных",
        [],
        ["статистик", "теория вероятност", "матстат", "распределени"],
    ),
    (
        "Анализ данных",
        "Анализ данных",
        ["Теория вероятностей и статистика"],
        ["анализ данных", "data science", "big data", "pandas", "sql"],
    ),
    (
        "Экспериментальные методы",
        "Экспериментальные",
        [],
        ["эксперимент", "лаборатор", "измерен", "лабораторн"],
    ),
    (
        "Спектроскопия и оптика",
        "Экспериментальные",
        ["Общая физика"],
        ["спектр", "оптик", "лазер", "фотон", "спектроскопи"],
    ),
    (
        "Микроскопия и материаловедение",
        "Экспериментальные",
        ["Общая физика"],
        ["микроскоп", "sem", "tem", "afm", "электронная микроскопи"],
    ),
    (
        "Общая физика",
        "Фундаментальная физика",
        [],
        ["физик", "механик", "электродинамик", "термодинамик"],
    ),
    (
        "Квантовая механика",
        "Фундаментальная физика",
        ["Общая физика"],
        ["квантов", "dft", "фотон", "волновая функци"],
    ),
    (
        "Физика конденсированного состояния",
        "Фундаментальная физика",
        ["Квантовая механика"],
        ["конденсирован", "кристалл", "фазов", "твёрдое тело"],
    ),
    (
        "Биоинформатика",
        "Биологические методы",
        ["Программирование на Python", "Анализ данных"],
        ["биоинформатик", "геном", "секвенировани", "ncbi", "rna"],
    ),
    (
        "Молекулярная биология",
        "Биологические методы",
        [],
        ["молекуляр", "днк", "рнк", "пцр", "генетик"],
    ),
    (
        "Аналитическая химия",
        "Химический синтез",
        [],
        ["аналитич", "титровани", "хроматограф", "масс-спектрометр"],
    ),
    (
        "Органический синтез",
        "Химический синтез",
        ["Аналитическая химия"],
        ["синтез", "органическ", "фармацевтик", "катализ"],
    ),
    (
        "Математическое моделирование",
        "Вычислительные",
        ["Численные методы"],
        ["моделировани", "симуляци", "идентификаци систем", "регресси"],
    ),
    (
        "Научная коммуникация",
        "Научная коммуникация",
        [],
        ["стать", "публикаци", "доклад", "конференци", "презентаци"],
    ),
    (
        "Управление научными проектами",
        "Управление проектами",
        ["Научная коммуникация"],
        ["управлен", "проект", "руковод", "команд", "планировани"],
    ),
    (
        "MATLAB и вычислительные пакеты",
        "Вычислительные",
        ["Программирование на Python"],
        ["matlab", "mathematica", "maple"],
    ),
    (
        "Радиационная физика",
        "Фундаментальная физика",
        ["Общая физика"],
        ["радиаци", "ядерн", "радиоактив", "ионизирующ"],
    ),
    (
        "Научное программирование (ROOT/GEANT4)",
        "Программирование",
        ["Программирование на C/C++", "Радиационная физика"],
        ["root", "geant4", "cern"],
    ),
    (
        "Химическая технология",
        "Химический синтез",
        ["Аналитическая химия"],
        ["технолог", "производств", "реактор", "промышлен"],
    ),
    (
        "Биомедицинская инженерия",
        "Инженерия",
        ["Молекулярная биология", "Экспериментальные методы"],
        ["биомедицин", "медицинск", "биосенсор", "протезировани"],
    ),
]


def _match_skill_to_courses(skill: str) -> list[tuple[str, str, list[str]]]:
    """Return list of (course_name, category, prereqs) that cover this skill."""
    results: list[tuple[str, str, list[str]]] = []
    skill_lower = skill.lower()

    for course_name, category, prereqs, expressions in _COURSE_CATALOGUE:
        for expr in expressions:
            try:
                if re.search(expr, skill_lower):
                    results.append((course_name, category, prereqs))
                    break
            except re.error:
                continue

    return results


def recommend_courses(
    skills: list[str],
    max_courses: int = 15,
) -> dict:
    """Recommend courses to teach a given skill set.

    Args:
        skills: List of skill lemmas from the market (e.g. from SkillFrequency).
        max_courses: Maximum number of courses to recommend.

    Returns:
        Dict with:
        - courses: list of (course_name, category, covered_skills, prerequisites)
        - coverage: fraction of input skills covered
        - uncovered_skills: skills not matched to any course
        - by_category: {category: list of course_names}
        - semester_plan: list of (semester, courses) respecting prerequisites
    """
    # Map each skill to candidate courses
    skill_courses: dict[str, list[tuple[str, str, list[str]]]] = {}
    for skill in skills:
        matches = _match_skill_to_courses(skill)
        if matches:
            skill_courses[skill] = matches

    # Count how many skills each course covers (for greedy selection)
    course_coverage: dict[str, tuple[str, list[str], list[str], set[str]]] = {}
    for skill, matches in skill_courses.items():
        for course_name, category, prereqs in matches:
            if course_name not in course_coverage:
                course_coverage[course_name] = (category, prereqs, [], set())
            _, _, _, skill_set = course_coverage[course_name]
            skill_set.add(skill)

    # Greedy: select courses that cover most additional skills
    selected: list[tuple[str, str, int, list[str], list[str]]] = []
    covered_skills: set[str] = set()
    remaining = dict(course_coverage)

    while remaining and len(selected) < max_courses:
        best_name: str | None = None
        best_gain = 0

        for name, (_, _, _, skill_set) in remaining.items():
            new_skills = skill_set - covered_skills
            if len(new_skills) > best_gain:
                best_gain = len(new_skills)
                best_name = name

        if best_name is None or best_gain == 0:
            break

        cat, prereqs, _, skill_set = remaining.pop(best_name)
        selected.append((best_name, cat, best_gain, prereqs, sorted(skill_set)))
        covered_skills.update(skill_set)

    # Compute semester plan (topological sort respecting prereqs)
    semester_plan: list[list[str]] = []
    scheduled: set[str] = set()
    course_dict = {name: prereqs for name, _, _, prereqs, _ in selected}

    while len(scheduled) < len(selected):
        semester: list[str] = []
        for name, _, _, prereqs, _ in selected:
            if name in scheduled:
                continue
            if all(p not in course_dict or p in scheduled for p in prereqs):
                semester.append(name)
        if not semester:
            # Cycle or orphan prerequisites — schedule remaining
            for name, _, _, _, _ in selected:
                if name not in scheduled:
                    semester.append(name)
        scheduled.update(semester)
        semester_plan.append(semester)

    # Group by category
    by_category: dict[str, list[str]] = {}
    for name, cat, _, _, _ in selected:
        by_category.setdefault(cat, []).append(name)

    uncovered = [s for s in skills if s not in skill_courses]

    return {
        "courses": [
            {
                "name": name,
                "category": cat,
                "skills_covered": skills,
                "skill_count": count,
                "prerequisites": prereqs,
            }
            for name, cat, count, prereqs, skills in selected
        ],
        "coverage": len(covered_skills) / len(skills) if skills else 0.0,
        "total_skills": len(skills),
        "covered_skills": len(covered_skills),
        "uncovered_skills": uncovered[:30],  # show at most 30
        "by_category": by_category,
        "semester_plan": [
            {"semester": i + 1, "courses": courses} for i, courses in enumerate(semester_plan)
        ],
    }


def recommend_curriculum_for_specialty(
    db: Database,
    specialty: str,
    top_n_skills: int = 30,
    max_courses: int = 15,
) -> dict:
    """Generate a full curriculum recommendation for a specialty.

    Args:
        db: Database instance.
        specialty: Specialty name (e.g. "physics", "stem-it").
        top_n_skills: How many top market skills to use.
        max_courses: Maximum number of courses to recommend.

    Returns:
        Dict with market_profile, course_recommendations, and semester_plan.
    """
    skill_freqs: list[SkillFrequency] = db.get_skill_frequencies(specialty, top_n=top_n_skills)
    if not skill_freqs:
        return {
            "specialty": specialty,
            "error": f"No skill data for '{specialty}'",
            "total_skills": 0,
            "recommendations": {},
        }

    skills = [sf.lemma for sf in skill_freqs[:top_n_skills]]
    course_result = recommend_courses(skills, max_courses=max_courses)

    return {
        "specialty": specialty,
        "total_skills": len(skills),
        "market_top_skills": skills[:top_n_skills],
        "recommendations": course_result,
    }


def curriculum_gap_analysis(
    db: Database,
    specialty: str,
    current_curriculum_skills: list[str],
    top_n_market: int = 30,
) -> dict:
    """Compare existing ITMO curriculum against market demands.

    Args:
        db: Database instance.
        specialty: Specialty name.
        current_curriculum_skills: Skills taught in current ITMO curriculum.
        top_n_market: How many top market skills to compare against.

    Returns:
        Dict with gap analysis: missing, covered, extra, coverage_pct.
    """
    skill_freqs = db.get_skill_frequencies(specialty, top_n=top_n_market)
    market_skills = {sf.lemma.lower() for sf in skill_freqs}
    taught_skills = {s.lower() for s in current_curriculum_skills}

    covered = market_skills & taught_skills
    missing = market_skills - taught_skills
    extra = taught_skills - market_skills

    coverage_pct = len(covered) / len(market_skills) * 100 if market_skills else 0.0

    return {
        "specialty": specialty,
        "market_skills": len(market_skills),
        "taught_skills": len(taught_skills),
        "covered": sorted(covered),
        "missing": sorted(missing),
        "extra_taught": sorted(extra),
        "coverage_pct": round(coverage_pct, 1),
        "recommendation": (
            f"Добавить {len(missing)} рыночных навыков: {', '.join(sorted(missing)[:10])}..."
            if len(missing) > 10 and missing
            else f"Добавить навыки: {', '.join(sorted(missing))}"
            if missing
            else "Учебный план полностью покрывает требования рынка"
        ),
    }
