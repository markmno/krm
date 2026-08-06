"""Statistical hypothesis testing for competency-role model discovery.

Provides statistical validation for discovered competency-role clusters:
- ClusterSignificanceTester: bootstrap silhouette and gap statistic for cluster validation
- ClusterStabilityTester: chi-squared, ANOVA, and bootstrap Rand index for stability
- GapAnalyzer: curriculum-market coverage gap analysis
"""

from hh_competency.stats.gaps import GapAnalyzer
from hh_competency.stats.significance import ClusterSignificanceTester
from hh_competency.stats.stability import ClusterStabilityTester

__all__ = [
    "ClusterSignificanceTester",
    "ClusterStabilityTester",
    "GapAnalyzer",
]
