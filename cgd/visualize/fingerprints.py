"""
cgd.visualize.fingerprints

实现 FingerprintPlotter 类，用于绘制行为指纹和效应指纹。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

class FingerprintPlotter:
    """
    一个用于创建行为指纹和效应指纹雷达图的类。
    """
    def __init__(self, labels: List[str]):
        """
        初始化绘图器。

        Args:
            labels (List[str]): 雷达图各个主轴的标签列表 (原子事件名称)。
        """
        self.K = len(labels)
        self.labels = labels
        self.angles = np.linspace(0, 2 * np.pi, self.K, endpoint=False).tolist()
        # 将第一个角度放在顶部
        self.angles = np.roll(self.angles, -int(self.K/4))

    def _setup_polar_axis(self, ax: plt.Axes, title: str):
        """配置雷达图的坐标轴。"""
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(self.angles)
        ax.set_xticklabels(self.labels)
        ax.set_rlabel_position(0)
        ax.set_title(title, weight='bold', size='large', position=(0.5, 1.1))
        ax.spines['polar'].set_visible(False) # 隐藏最外圈
        ax.grid(color='grey', linestyle='--', linewidth=0.5)

    def plot_behavior_fingerprint(
        self, 
        p_vector: np.ndarray, 
        ax: plt.Axes,
        label: str = 'Behavior',
        color: str = 'blue',
        alpha: float = 0.25
    ):
        """
        绘制一个行为指纹。

        Args:
            p_vector (np.ndarray): 概率分布向量。
            ax (plt.Axes): 要绘制在其上的 matplotlib polar axes。
            label (str): 图例标签。
            color (str): 指纹的颜色。
            alpha (float): 填充的透明度。
        """
        if len(p_vector) != self.K:
            raise ValueError(f"数据维度 ({len(p_vector)}) 与标签维度 ({self.K}) 不匹配。")
        
        # 为了闭合图形，需要将第一个点的数据追加到末尾
        values = np.concatenate((p_vector, [p_vector[0]]))
        angles = np.concatenate((self.angles, [self.angles[0]]))
        
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=alpha)
        ax.set_ylim(0, 1)

    def plot_effect_fingerprint(
        self, 
        v_eigen: np.ndarray, 
        ax: plt.Axes,
        label: str = 'Effect',
        color_positive: str = 'red',
        color_negative: str = 'blue',
        alpha: float = 0.3
    ):
        """
        绘制一个效应指纹。

        Args:
            v_eigen (np.ndarray): 本征效应向量。
            ax (plt.Axes): 要绘制在其上的 matplotlib polar axes。
            label (str): 图例标签。
            color_positive (str): 正向效应的颜色。
            color_negative (str): 负向效应的颜色。
            alpha (float): 填充的透明度。
        """
        if len(v_eigen) != self.K:
            raise ValueError(f"向量维度 ({len(v_eigen)}) 与标签维度 ({self.K}) 不匹配。")
            
        # 效应指纹以“赤道”为中心，需要对数据进行缩放和转换
        # 这里我们用一个简单的线性映射，更复杂的可以自定义
        max_abs_val = np.max(np.abs(v_eigen))
        if max_abs_val == 0: # 如果是零向量
            scaled_values = np.zeros(self.K)
        else:
            scaled_values = v_eigen / max_abs_val * 0.5 # 缩放到 [-0.5, 0.5]
        
        plot_values = scaled_values + 0.5 # 移动到 [0, 1] 区间，赤道在 0.5

        # 闭合图形
        values = np.concatenate((plot_values, [plot_values[0]]))
        angles = np.concatenate((self.angles, [self.angles[0]]))
        
        # 绘制赤道
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 0.5), color='black', linestyle='--', linewidth=1, label='Equator (Zero Effect)')

        # 绘制效应轮廓
        ax.plot(angles, values, color='black', linewidth=1.5, label=label)
        
        # 差异化着色
        ax.fill_between(angles, 0.5, values, where=values > 0.5, color=color_positive, alpha=alpha, interpolate=True)
        ax.fill_between(angles, 0.5, values, where=values < 0.5, color=color_negative, alpha=alpha, interpolate=True)
        
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['-Max', '-Half', 'Zero', '+Half', '+Max'])
