# cgd/dynamics/potential.py (修复版)

"""
cgd.dynamics.potential

实现CGD标准物理学（高斯模型）势能计算的NumPy实现。
"""

from typing import List
import numpy as np

# --- 核心依赖 ---
from ..core.source import GravitationalSource
from ..core.universe import Universe
from ..geometry.simplex import distance_FR # <-- 确保这里导入的是NumPy版的distance_FR
from ..geometry.transformations import exp_map_from_origin

def potential_gaussian(p: np.ndarray, sources: List[GravitationalSource], universe: Universe) -> float:
    """
    计算在点 p 的总高斯势能 (NumPy版本)。
    """
    total_potential = 0.0
    
    K = universe.K
    alpha = universe.alpha
    radius = universe.simplex.radius
    
    if radius == 0:
        return 0.0

    for source in sources:
        p_stim = exp_map_from_origin(source.v_eigen)
        
        # --- 确保调用的是NumPy版的距离函数 ---
        dist = distance_FR(p, p_stim)
        relative_dist = dist / radius
        
        potential_s = -source.strength * np.exp(-((alpha * relative_dist)**2))
        total_potential += potential_s
        
    return total_potential