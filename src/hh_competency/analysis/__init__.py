"""Analysis engine: skill frequencies, cross-specialty comparison, role clustering.

Public API
----------
- SkillAnalyzer: Extract and rank skills from vacancy descriptions.
- SkillComparator: Compare skill profiles across specialties.
- SkillClusterer: Discover role profiles via co-occurrence clustering.
- NoDataError: Raised when no vacancy data exists for a specialty.
"""

from hh_competency.analysis.clustering import SkillClusterer
from hh_competency.analysis.compare import SkillComparator
from hh_competency.analysis.skills import NoDataError, SkillAnalyzer

__all__ = [
    "NoDataError",
    "SkillAnalyzer",
    "SkillClusterer",
    "SkillComparator",
]
