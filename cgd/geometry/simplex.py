# cgd/geometry/simplex.py
"""
cgd.geometry.simplex

提供与概率单纯形几何属性相关的函数和类。
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from functools import lru_cache

# ----------------------------------------------------
# 独立的、可被外部导入的几何函数
# ----------------------------------------------------

@lru_cache(maxsize=32)  # 为常用K值缓存结果，增加缓存大小
def get_chaos_origin(K: int) -> np.ndarray:
    """
    返回维度 K 的混沌原点 O (均匀分布)。

    Args:
        K (int): 宇宙的维度。

    Returns:
        np.ndarray: 一个(K,)维的numpy数组，所有元素均为 1/K。
    """
    if not isinstance(K, int) or K <= 0:
        raise ValueError(f"维度 K 必须为正整数，但收到了 {K}。")
    return np.full(K, 1.0 / K)

@lru_cache(maxsize=32)
def get_radius(K: int) -> float:
    """
    返回维度 K 的宇宙最大半径 D_max(K)。

    Args:
        K (int): 宇宙的维度。

    Returns:
        float: 从混沌原点到任一顶点的测地线距离。
    """
    if not isinstance(K, int) or K <= 0:
        raise ValueError(f"维度 K 必须为正整数，但收到了 {K}。")
    if K == 1:
        return 0.0
    return 2 * np.arccos(np.sqrt(1.0 / K))

def distance_FR(p: np.ndarray, q: np.ndarray) -> float:
    """
    计算两个概率分布 p 和 q 之间的费雪-饶距离。

    Args:
        p (np.ndarray): 第一个概率向量。
        q (np.ndarray): 第二个概率向量。

    Returns:
        float: 两点之间的测地线距离。
    """
    # 确保输入是 numpy 数组并进行数据清洗
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    if p.shape != q.shape:
        raise ValueError(f"输入向量的形状不匹配: {p.shape} vs {q.shape}")
    
    # 确保向量被归一化
    p /= np.sum(p)
    q /= np.sum(q)

    # 防止因浮点数误差导致 sqrt 输入为负
    p[p < 0] = 0
    q[q < 0] = 0

    # 计算点积
    dot_product = np.sum(np.sqrt(p * q))

    # 防止因浮点数误差导致 arccos 输入大于1
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    distance = 2 * np.arccos(dot_product)
    return distance


# ----------------------------------------------------
# Simplex 类作为上述函数的便捷封装器
# ----------------------------------------------------

@dataclass(frozen=True)
class Simplex:
    """
    一个存储特定维度 K 的概率单纯形几何常数的便捷数据类。
    
    它作为 Universe 类的一个属性，提供了对与维度相关的几何常数的
    快速访问，并利用缓存避免重复计算。

    Attributes:
        K (int): 宇宙的维度 (原子事件的数量)。
    """
    K: int

    @property
    def origin(self) -> np.ndarray:
        """混沌原点 O 的坐标。"""
        return get_chaos_origin(self.K)

    @property
    def radius(self) -> float:
        """宇宙的最大半径 D_max(K)。"""
        return get_radius(self.K)

    def __repr__(self):
        return f"Simplex(K={self.K}, radius={self.radius:.4f})"