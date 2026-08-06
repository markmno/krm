"""Dataset builder: filter HF CSV → science vacancies JSONL per specialty.

Separate filtering layers:
  1. IT vacancy exclusion — remove pure IT jobs by title
  2. Science matching — keep only science/engineering vacancies
  3. Category classification — assign to physics/biology/chemistry/ecology/engineering
  4. Skill buzzwords — filtered later in NLP phase, NOT here
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Iterator

# ── IT Exclusion Rules (vacancy title) ──────────────────────────────

_EXCLUDE_TITLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Pure IT roles
        r"\bпрограммист\b",
        r"\bdev(eloper)?\b",
        r"\bdevops\b",
        r"\bqa\b",
        r"\bтестировщик\b",
        r"\bдизайнер\b",
        r"\bdesigner\b",
        r"\bмаркетолог\b",
        r"\bmarketing\b",
        r"\bseo\b",
        r"\bsmm\b",
        r"\bпродавец\b",
        r"\bменеджер по продажам\b",
        r"\bрекрутер\b",
        r"\bhr\b",
        r"\bкадровик\b",
        r"\bюрист\b",
        r"\bбухгалтер\b",
        r"\bфинансист\b",
        r"\bэкономист\b",
        r"\bлогист\b",
        r"\bкладовщик\b",
        r"\bводитель\b",
        r"\bкурьер\b",
        r"\bохранник\b",
        r"\bуборщик\b",
        r"\bповар\b",
        r"\bадминистратор\b(?!.*(?:баз|систем|сети|лаборатор))",
        r"\bоператор\b(?!.*(?:установк|прибор|лаборатор|спектр|микроскоп|реактор|станк|чпу))",
        r"\bменеджер\b(?!.*(?:проект|продукт|r&d|лаборатор|научн|исследова))",
        # Support roles that are clearly non-science
        r"\bсекретарь\b",
        r"\bпомощник руководителя\b",
        r"\bофис-менеджер\b",
    ]
]

# ── Science Matching Rules (vacancy title + description) ────────────

_SCIENCE_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "physics",
        re.compile(
            r"физик|физич|квантов|ядерн|радиаци|"
            r"ускорител|спектроскопи|лазер|оптик|фотоник|"
            r"астрофизик|геофизик|биофизик|теплофизик|"
            r"плазм|дозиметр|рентген|магнитн|сверхпровод|"
            r"конденсирован|электродинамик|термодинамик",
            re.IGNORECASE,
        ),
    ),
    (
        "chemistry",
        re.compile(
            r"хими|аналитич.*(?:лаборатор|хими|контрол)|"
            r"органический синтез|неорганическ|фармацевт|фармаколог|"
            r"катализ|полимер|хроматограф|спектрометр|"
            r"нефтехим|газохим|биохим|геохим|радиохим|электрохим|"
            r"масс-спектр|титровани|реактив",
            re.IGNORECASE,
        ),
    ),
    (
        "biology",
        re.compile(
            r"биолог|генетик|микробиолог|биоинформатик|молекулярн.*(?:биолог|генетик)|"
            r"биотехнолог|вирусолог|иммунолог|токсиколог|"
            r"днк|рнк|пцр|секвенирован|геном|протеом|"
            r"клеточн|цитолог|гистолог|эмбриолог|"
            r"ботаник|зоолог|энтомолог|орнитолог|ихтиолог|"
            r"эколог|природоохран|биоразнообраз",
            re.IGNORECASE,
        ),
    ),
    (
        "ecology",
        re.compile(
            r"эколог|природоохран|окружающ.*(?:сред|природ)|"
            r"климатолог|метеоролог|гидролог|океанолог|почвовед|"
            r"геоэкологи|устойчив.*развити|отход|рециклинг|"
            r"зелен|возобновляем|энергоэффективн|углеродн.*(?:след|единиц)",
            re.IGNORECASE,
        ),
    ),
    (
        "engineering",
        re.compile(
            r"инженер(?!.*(?:программ|software|тестиров|qa\b|devops))|"
            r"конструктор|проектировщик|метролог|технолог|"
            r"расчётчик|чертёжник|схемотехник|"
            r"техническ.*специалист|техническ.*эксперт|"
            r"механик|электроник|автоматизаци.*(?:производ|технолог)|"
            r"кипиа|асу.*тп|робототехник|мехатроник|"
            r"материаловед|теплотехник|энергетик|электротехник",
            re.IGNORECASE,
        ),
    ),
]


def _is_science_vacancy(title: str, description: str) -> list[str]:
    """Check if vacancy matches any science category.

    Returns list of matching category names (empty = not science).
    """
    text = f"{title} {description}"
    matched: list[str] = []
    for category, pattern in _SCIENCE_RULES:
        if pattern.search(text):
            matched.append(category)
    return matched


def _is_pure_it(title: str) -> bool:
    """Check if vacancy title is a pure IT job that should be excluded."""
    for pattern in _EXCLUDE_TITLE_PATTERNS:
        if pattern.search(title):
            return True
    return False


def build_science_dataset(
    csv_path: str | Path,
    output_dir: str | Path,
    min_description_len: int = 50,
) -> dict[str, int]:
    """Build filtered science vacancy JSONL files from HF CSV.

    Args:
        csv_path: Path to HF dataset CSV (semicolon-delimited).
        output_dir: Directory to write per-specialty JSONL files.
        min_description_len: Minimum description length to keep.

    Returns:
        Dict with stats: {specialty: vacancy_count}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    writers: dict[str, TextIO] = {}
    total_read = 0
    total_excluded_it = 0
    total_not_science = 0
    total_short_desc = 0

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')

        for row in reader:
            total_read += 1

            title = row.get("название_вакансии", "")
            description = row.get("требования_обязанности", "")

            if not title:
                continue

            # Layer 1: IT exclusion
            if _is_pure_it(title):
                total_excluded_it += 1
                continue

            # Layer 2: Science matching
            categories = _is_science_vacancy(title, description)
            if not categories:
                total_not_science += 1
                continue

            # Layer 3: Minimum description
            if len(description) < min_description_len:
                total_short_desc += 1
                continue

            # Parse salary: "зарплата_число" is a numeric column
            salary_raw = row.get("зарплата_число", "")
            salary_val = None
            if salary_raw and salary_raw.strip():
                try:
                    salary_val = int(float(salary_raw))
                except (ValueError, TypeError):
                    pass

            vacancy = {
                "id": f"csv_{total_read}",
                "name": title,
                "description": description,
                "key_skills": [],
                "salary": {
                    "from": salary_val,
                    "to": None,
                    "currency": "RUR",
                },
                "experience": {"id": row.get("опыт работы", ""), "name": row.get("опыт работы", "")},
                "area": {"id": "", "name": row.get("местоположение", "")},
                "professional_roles": [],
                "employer": {"id": "", "name": row.get("название_компании", "")},
                "published_at": "",
            }

            # Write to each matching category's JSONL
            for cat in categories:
                if cat not in writers:
                    path = output_dir / f"{cat}.jsonl"
                    writers[cat] = open(path, "a", encoding="utf-8-sig")
                    counts[cat] = 0

                writers[cat].write(json.dumps(vacancy, ensure_ascii=False) + "\n")
                counts[cat] += 1

    # Close all writers
    for w in writers.values():
        w.close()

    print(f"Read: {total_read}")
    print(f"Excluded (IT): {total_excluded_it}")
    print(f"Excluded (not science): {total_not_science}")
    print(f"Excluded (short desc): {total_short_desc}")
    print(f"Kept: {sum(counts.values())}")
    for cat, cnt in sorted(counts.items()):
        print(f"  {cat}: {cnt}")

    return counts


def _yield_csv_rows(csv_path: str | Path) -> Iterator[dict[str, str]]:
    """Yield rows from semicolon-delimited CSV without loading entire file."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        yield from reader
