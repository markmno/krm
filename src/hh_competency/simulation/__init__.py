"""Monte Carlo simulation, curriculum optimization, and graduate-market matching.

Public API:
    MonteCarloEngine — statistical simulation of curriculum-vacancy fit.
    CurriculumOptimizer — greedy optimization of skill sets for market coverage.
    GraduateCurriculum — skills a graduate has from their ITMO curriculum.
    GraduatePool — collection of graduates with different specializations.
    simulate_graduate_match — match graduates against real market roles.
"""

from hh_competency.simulation.graduate_matcher import (
    GraduateCurriculum,
    GraduatePool,
    simulate_graduate_match,
)
from hh_competency.simulation.monte_carlo import MonteCarloEngine
from hh_competency.simulation.optimizer import CurriculumOptimizer

__all__ = [
    "CurriculumOptimizer",
    "GraduateCurriculum",
    "GraduatePool",
    "MonteCarloEngine",
    "simulate_graduate_match",
]
