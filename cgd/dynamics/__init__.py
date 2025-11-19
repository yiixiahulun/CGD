"""
cgd.dynamics

实现CGD标准物理学（高斯模型）的核心计算引擎。
"""

from .force import calculate_F_net
from .potential import potential_gaussian
from .solvers import EquilibriumFinder, TrajectorySimulator

__all__ = [
    "potential_gaussian",
    "calculate_F_net",
    "EquilibriumFinder",
    "TrajectorySimulator",
]
