# cgd/visualize/landscape.py
"""
实现 LandscapePlotter 类，用于绘制势能地貌和合力场图。
主要用于 K=3 的情况。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import ternary # 导入 ternary 库

# 从我们的其他模块中导入所需的对象和工具
from ..core.source import GravitationalSource
from ..core.universe import Universe
from ..dynamics.potential import potential_gaussian
from ..dynamics.force import calculate_F_net

class LandscapePlotter:
    """
    一个用于绘制二维单纯形（K=3）上的势能地貌和合力场图的类。
    """
    def __init__(self, universe: Universe, sources: List[GravitationalSource]):
        if universe.K != 3:
            raise ValueError("LandscapePlotter 目前仅支持 K=3 的宇宙。")
        self.universe = universe
        self.sources = sources
        self.scale = 100  # 网格密度

    def plot_potential_surface(self, ax: plt.Axes, **kwargs) -> ternary.TernaryAxesSubplot:
        """
        绘制势能地貌的热力图。

        Args:
            ax (plt.Axes): matplotlib axes 对象。
            **kwargs: 传递给 `tax.heatmapf` 的其他参数 (例如 cmap, style)。

        Returns:
            ternary.TernaryAxesSubplot: 创建的 ternary axes 对象，以便后续叠加绘图。
        """
        fig = ax.figure
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=self.scale)
        
        # --- 基本绘图设置 ---
        tax.boundary(linewidth=1.5)
        tax.gridlines(color="white", multiple=self.scale/10, linewidth=0.5, alpha=0.5)
        tax.get_axes().axis('off')
        tax.clear_matplotlib_ticks()

        # --- 核心修正：使用 `heatmapf` 替代 `contourf` ---
        # 定义计算势能的函数
        def potential_wrapper(p):
            # ternary库给出的p是(x,y)坐标，需要转换回概率向量
            # p[0] is fraction of B, p[1] is fraction of L for a (L,B,R) simplex
            # 我们的约定是 (p0, p1, p2)，对应 (Right, Left, Bottom)
            p_vec = np.array([p[1], p[0], self.scale - p[0] - p[1]]) / self.scale
            p_vec /= np.sum(p_vec)
            return potential_gaussian(p_vec, self.sources, self.universe)

        # 使用 heatmapf 绘制填充的热力图
        # nlevels 参数在 heatmapf 中可能不直接支持，但 cmap 等参数可以
        # 如果需要类似等高线的效果，可以叠加 tax.heatmap 和 tax.contour
        tax.heatmapf(potential_wrapper, scale=self.scale, **kwargs)
        
        # --- 绘图收尾工作 ---
        tax.ticks(axis='lbr', linewidth=1, multiple=self.scale/5, tick_formats="%.1f")
        tax.set_title("Potential Landscape", fontsize=14, pad=20)
        
        return tax

    def plot_force_field(self, tax: ternary.TernaryAxesSubplot, **kwargs):
        """
        在已有的地貌图上叠加合力场流线图。

        Args:
            tax (ternary.TernaryAxesSubplot): ternary axes 对象。
            **kwargs: 传递给 `tax.streamplot` 的其他参数。
        """
        
        def force_wrapper(p):
            # (L, B, R) -> (p1, p0, p2)
            p_vec = np.array([p[1], p[0], self.scale - p[0] - p[1]]) / self.scale
            p_vec /= np.sum(p_vec)
            f_vec = calculate_F_net(p_vec, self.sources, self.universe)
            
            # 将力从单纯形坐标(f0, f1, f2)转换回ternary的(x,y)坐标系
            # (f_R, f_L, f_B) -> (fx, fy)
            # 这里的转换基于ternary库的内部坐标系统
            # fx is horizontal (related to p1 vs p2)
            # fy is vertical (related to p0)
            fx = (f_vec[1] - f_vec[2]) / 2.0
            fy = (2 * f_vec[0] - f_vec[1] - f_vec[2]) / (2 * np.sqrt(3))
            return (fx, fy)

        tax.streamplot(force_wrapper, scale=self.scale, **kwargs)
        tax.set_title("Potential & Force Field", fontsize=14, pad=20)