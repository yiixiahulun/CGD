# cgd/measure/purification.py
"""
cgd.measure.purification

提供标准的测量函数，用于从实验数据中计算并构建 GravitationalSource 对象。
这是连接实验观测和CGD理论实体的核心接口。
"""

from typing import Literal, List
import numpy as np

# 从我们的几何工具箱中导入核心映射工具
from ..geometry.transformations import log_map_from_origin
# 从核心对象模块导入 GravitationalSource，以便我们可以创建它
from ..core.source import GravitationalSource


def measure_source(
    name: str,
    p_total: np.ndarray,
    p_stage: np.ndarray,
    event_labels: List[str],
    measurement_type: Literal['absolute', 'substitution'] = 'substitution'
) -> GravitationalSource:
    """
    通过几何净化，从实验数据中测量一个引力源，并返回一个完整的对象。

    此函数是连接实验观测和CGD理论实体的核心接口。它执行几何净化
    (v_total - v_stage)，并根据指定的测量类型和坐标系标签，返回一个
    信息完备的 GravitationalSource 对象。

    Args:
        name (str): 
            要创建的引力源的名称 (e.g., "Banana_Scent", "Drug_X_vs_Vehicle")。
        p_total (np.ndarray): 
            在“单源净化实验”中观测到的总效应概率分布。
        p_stage (np.ndarray): 
            在“情景校准实验”中观测到的舞台基线概率分布。
        event_labels (List[str]):
            一个包含原子事件标签的列表，用于定义本次测量的坐标系。
            其长度必须与 p_total 和 p_stage 的维度相匹配。
        measurement_type (Literal['absolute', 'substitution'], optional): 
            指定本次测量的理论类型。默认为 'substitution'，对应于
            绝大多数标准对比实验的场景。

    Returns:
        GravitationalSource: 
            一个封装了测量结果的、信息完备的引力源对象。
    """
    K = len(event_labels)
    # --- 增强的验证 ---
    if len(p_total) != K or len(p_stage) != K:
        raise ValueError(
            f"p_total (维度 {len(p_total)})、p_stage (维度 {len(p_stage)}) "
            f"和 event_labels (数量 {K}) 的维度必须完全匹配。"
        )

    # 1. 后台状态解码
    v_total = log_map_from_origin(p_total)
    v_stage = log_map_from_origin(p_stage)

    # 2. 后台效应分离
    v_eigen = v_total - v_stage
    
    # 3. 构建并返回一个完整的、带有“记忆”的物理对象
    return GravitationalSource(
        name=name, 
        v_eigen=v_eigen, 
        v_type=measurement_type,
        labels=tuple(event_labels)
    )


# --- (可选但强烈推荐) 保留旧函数名作为别名，以实现向后兼容 ---
def geometric_purification(p_total: np.ndarray, p_stage: np.ndarray) -> np.ndarray:
    """
    【旧版接口，已废弃】执行纯粹的几何净化数学运算。

    警告：此函数已被新的、更安全的 `measure_source` 函数取代。
    它只返回一个裸的 numpy 数组，丢失了重要的 v_type 和 labels 信息，
    在 v3.0 及以后的框架中极易导致错误。仅为向后兼容性保留。

    Returns:
        np.ndarray: 被测量出的 v_eigen (裸数组)。
    """
    import warnings
    warnings.warn(
        "`geometric_purification` 已被 `measure_source` 取代。"
        "新函数会返回一个信息完备的 GravitationalSource 对象，更加安全。",
        DeprecationWarning,
        stacklevel=2
    )
    if len(p_total) != len(p_stage):
        raise ValueError("p_total 和 p_stage 的维度必须相同。")
        
    v_total = log_map_from_origin(p_total)
    v_stage = log_map_from_origin(p_stage)
    return v_total - v_stage