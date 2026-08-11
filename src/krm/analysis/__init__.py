"""KRM Analysis Subpackage.

Post-pipeline statistical analysis: trends, cross-specialty comparison,
cluster validation, axis discovery, and role archetype classification.

All modules accept pandas DataFrames or Python primitives — no database dependency.
"""

from krm.analysis.trends import analyze_salary_trends, analyze_trends, linear_slope, mann_kendall
from krm.analysis.compare import SkillComparator
from krm.analysis.statistical import ClusterSignificanceTester
from krm.analysis.stability import ClusterStabilityTester
from krm.analysis.axis_discovery import AxisDiscoveryEngine, AxisDiscoveryResult
from krm.analysis.archetypes import RoleArchetype, classify_archetypes
from krm.analysis.differentiating import collect_differentiating_skills
from krm.analysis.temporal_consistency import (  # noqa: I001
    block_bootstrap_ci,
    breslow_day_test,
    cochran_armitage_test,
    cochran_mantel_haenszel_test,
    fleiss_kappa_agreement,
    monte_carlo_stability,
    permutation_stability_test,
    temporal_stability_report,
    validate_characteristics,
)

__all__ = [
    # trends
    "analyze_salary_trends",
    "analyze_trends",
    "linear_slope",
    "mann_kendall",
    # compare
    "SkillComparator",
    # statistical
    "ClusterSignificanceTester",
    # stability
    "ClusterStabilityTester",
    # axis_discovery
    "AxisDiscoveryEngine",
    "AxisDiscoveryResult",
    # archetypes
    "RoleArchetype",
    "classify_archetypes",
    # differentiating
    "collect_differentiating_skills",
    # temporal_consistency
    "block_bootstrap_ci",
    "breslow_day_test",
    "cochran_armitage_test",
    "cochran_mantel_haenszel_test",
    "fleiss_kappa_agreement",
    "monte_carlo_stability",
    "permutation_stability_test",
    "temporal_stability_report",
    "validate_characteristics",
]
