# cgd/measure/discovery.py
"""
cgd.measure.discovery

实现 ProbeFitter 类，用于从理论预测与现实观测的偏差中发现新的探针。
这是CGD理论自我进化的核心引擎。
"""

from typing import List

import numpy as np
from scipy.optimize import minimize

# --- 从我们库的其他部分导入必要的对象和函数 ---
from ..core.source import GravitationalSource
from ..geometry.simplex import distance_FR
from ..geometry.transformations import exp_map_from_origin, log_map_from_origin


class ProbeFitter:
    """
    一个用于从残差中拟合探针，并返回一个完整 GravitationalSource 对象的类。
    """

    def __init__(
        self, p_predicted: np.ndarray, p_observed: np.ndarray, event_labels: List[str]
    ):
        """
        初始化 ProbeFitter。

        Args:
            p_predicted (np.ndarray): 理论模型给出的预测概率分布。
            p_observed (np.ndarray): 真实的实验观测概率分布。
            event_labels (List[str]): 定义了当前坐标系的原子事件标签列表。
        """
        if len(p_predicted) != len(p_observed) or len(p_predicted) != len(event_labels):
            raise ValueError(
                "p_predicted, p_observed 和 event_labels 的维度必须完全匹配。"
            )

        self.p_predicted = p_predicted
        self.p_observed = p_observed
        self.event_labels = tuple(event_labels)
        self.K = len(event_labels)

        # 将预测点和观测点都解码到 v 空间
        self.v_predicted = log_map_from_origin(self.p_predicted)
        self.v_observed = log_map_from_origin(self.p_observed)

    def _objective_function(self, v_probe_flat: np.ndarray) -> float:
        """
        目标函数，用于最小化加入探针后的预测与观测之间的几何距离。
        """
        # 确保探针向量的和为零
        v_probe = v_probe_flat - np.mean(v_probe_flat)

        # 在 LESA 框架下计算新预测：v_new = v_predicted + v_probe
        # 这里的 v_predicted 已经是我们解码好的基准
        v_new_predicted = self.v_predicted + v_probe
        p_new_predicted = exp_map_from_origin(v_new_predicted)

        # 返回与真实观测的距离
        return distance_FR(p_new_predicted, self.p_observed)

    def fit(self, probe_name: str = "Probe") -> GravitationalSource:
        """
        执行拟合，找到最佳的探针效应向量，并将其作为一个 GravitationalSource 对象返回。

        Args:
            probe_name (str, optional):
                为新发现的探针指定的名称。默认为 "Probe"。

        Returns:
            GravitationalSource:
                一个代表了被发现的探针的、信息完备的引力源对象。
                探针的 v_type 被定义为 'absolute'，因为它解释的是绝对偏差。
        """
        # 初始猜测：残差力向量 (v_observed - v_predicted) 是一个绝佳的起点
        initial_guess = self.v_observed - self.v_predicted

        result = minimize(
            self._objective_function,
            x0=initial_guess,
            method="BFGS",
            options={"gtol": 1e-9},
        )

        if result.success:
            # 从优化结果中提取最终的 v_eigen，并再次确保其和为零
            v_probe_optimal = result.x - np.mean(result.x)

            # 构建并返回一个完整的 GravitationalSource 对象
            return GravitationalSource(
                name=probe_name,
                v_eigen=v_probe_optimal,
                v_type="absolute",  # 探针解释的是绝对偏差，其本身是一个'absolute'类型的效应
                labels=self.event_labels,
            )
        else:
            raise RuntimeError(f"探针拟合未能收敛: {result.message}")
