"""
cgd.dynamics.force

实现合力的计算，通过对势能函数进行数值求导（梯度）。
"""

from typing import List
import numpy as np

# 从我们的其他模块中导入所需的对象和工具
from ..core.source import GravitationalSource
from ..core.universe import Universe
from .potential import potential_gaussian

def calculate_F_net(p: np.ndarray, sources: List[GravitationalSource], universe: Universe, h: float = 1e-6) -> np.ndarray:
    """
    通过数值方法（中心差分）计算在点 p 的合力 F_net（势能的负梯度）。

    Args:
        p (np.ndarray): 当前的概率分布点。
        sources (List[GravitationalSource]): 宇宙中的引力源列表。
        universe (Universe): 定义了物理学 (K, alpha) 的宇宙对象。
        h (float): 用于计算有限差分的极小步长。

    Returns:
        np.ndarray: 在点 p 的合力向量 F_net，它是一个切向量（和为0）。
    """
    K = universe.K
    grad = np.zeros(K)

    # 1. 计算每个方向上的偏导数
    for i in range(K):
        p_fwd = p.copy()
        p_bwd = p.copy()
        
        p_fwd[i] += h
        p_bwd[i] -= h
        
        # 必须重新归一化以保持在单纯形上（虽然步长很小，但这是严谨的做法）
        p_fwd /= np.sum(p_fwd)
        p_bwd /= np.sum(p_bwd)
        
        potential_fwd = potential_gaussian(p_fwd, sources, universe)
        potential_bwd = potential_gaussian(p_bwd, sources, universe)
        
        grad[i] = (potential_fwd - potential_bwd) / (2 * h)
        
    # 2. 将梯度向量投影到和为零的切空间
    grad_projected = grad - np.mean(grad)
    
    # 3. 力是势能的负梯度
    force_net = -grad_projected
    
    return force_net
