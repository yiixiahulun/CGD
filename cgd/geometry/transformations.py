# cgd/geometry/transformations.py

"""
cgd.geometry.transformations

提供在概率单纯形与“效应空间”(T_O)之间进行转换的核心几何映射函数。

注意：本模块只实现相对于混沌原点 O 的映射，因为这是CGD理论的核心。
"""
import numpy as np
from .simplex import get_chaos_origin

# 定义一个极小值 epsilon 用于数值稳定性
EPSILON = 1e-12

def log_map_from_origin(p: np.ndarray) -> np.ndarray:
    """
    计算从混沌原点 O 指向 p 的对数映射（效应向量 v）。
    v = Log_O(p)

    Args:
        p (np.ndarray): 终点概率向量。

    Returns:
        np.ndarray: 一个位于效应空间 T_O 的向量 v。
    """
    p = np.asarray(p, dtype=float)
    if not np.isclose(np.sum(p), 1.0) or np.any(p < 0):
        raise ValueError("输入向量 p 必须是一个有效的概率分布 (和为1，无负值)。")

    K = len(p)
    origin = get_chaos_origin(K)

    # 计算夹角的余弦值
    dot_product = np.sum(np.sqrt(origin * p))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 角度 theta 等于 D_FR(O, p) / 2
    theta = np.arccos(dot_product)
    
    # 如果两点重合，切向量为零向量
    if theta < EPSILON:
        return np.zeros(K)
        
    # 计算从 p 到 O 在球面上的投影向量
    u_raw = np.sqrt(p) - dot_product * np.sqrt(origin)
    
    norm_u = np.linalg.norm(u_raw)
    
    if norm_u < EPSILON:
        return np.zeros(K)

    # 最终的切向量 v (这里的坐标变换 / np.sqrt(origin) 是关键)
    v = (2 * theta / norm_u) * u_raw / np.sqrt(origin)
    
    # 强制确保和为零，消除浮点数累积误差
    return v - np.mean(v)

def exp_map_from_origin(v: np.ndarray) -> np.ndarray:
    """
    计算从混沌原点 O 出发，沿效应向量 v 行进的指数映射。
    p = Exp_O(v)

    Args:
        v (np.ndarray): 位于效应空间 T_O 的向量。

    Returns:
        np.ndarray: 沿测地线行进后的终点概率向量。
    """
    v = np.asarray(v, dtype=float)
    if not np.isclose(np.sum(v), 0.0):
        raise ValueError("输入向量 v 必须在切空间中 (和为0)。")

    K = len(v)
    origin = get_chaos_origin(K)

    # 首先，将效应向量 v 从单纯形坐标转换到球面坐标
    u = v * np.sqrt(origin)
    norm_u = np.linalg.norm(u)
    
    if norm_u < EPSILON:
        return origin

    # 测地线方程的平方根表示
    sqrt_p = np.cos(norm_u / 2.0) * np.sqrt(origin) + np.sin(norm_u / 2.0) * (u / norm_u)
    
    # 从平方根恢复概率
    p = sqrt_p**2
    return p / np.sum(p) # 加上归一化以确保数值精度