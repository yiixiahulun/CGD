"""
cgd.visualize.atlas

实现 AtlasPlotter 类，用于绘制 alpha 扫描后得到的宇宙图集。
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class AtlasPlotter:
    """
    一个用于绘制CGD宇宙图集的类。
    """

    def __init__(self, scan_results: Dict[str, Any]):
        """
        初始化绘图器。

        Args:
            scan_results (Dict[str, Any]):
                一个包含了 alpha 扫描结果的字典。
                例如: {
                    'alphas': np.array([...]),
                    'n_attractors': np.array([...]),
                    'lesa_deviation': np.array([...]),
                    'drift_trajectory_pca': np.array([...])
                }
        """
        self.results: Dict[str, Any] = scan_results
        self.alphas: NDArray[np.float64] = self.results["alphas"]

    def plot_phase_diagram(self, ax: plt.Axes) -> None:
        """绘制宇宙相图 (alpha vs. 吸引子数量)。"""
        n_attractors: NDArray[np.int_] = self.results["n_attractors"]
        ax.step(self.alphas, n_attractors, where="post")
        ax.set_xlabel("Alpha (α)")
        ax.set_ylabel("Number of Attractors")
        ax.set_title("Universe Phase Diagram")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_yticks(np.unique(n_attractors).astype(int))

    def plot_lesa_deviation(self, ax: plt.Axes) -> None:
        """绘制LESA偏差图 (alpha vs. 偏差)。"""
        deviation: NDArray[np.float64] = self.results["lesa_deviation"]
        ax.plot(self.alphas, deviation)
        ax.set_xlabel("Alpha (α)")
        ax.set_ylabel("LESA Deviation (D_FR)")
        ax.set_title("LESA Deviation Plot")
        ax.grid(True, linestyle="--", alpha=0.6)

    def plot_drift_trajectory(self, ax: plt.Axes) -> None:
        """绘制漂移轨迹图 (PCA投影)。"""
        pca_coords: NDArray[np.float64] = self.results["drift_trajectory_pca"]
        # pca_coords 应该是一个 (N, 2) 的数组
        sc = ax.scatter(
            pca_coords[:, 0], pca_coords[:, 1], c=self.alphas, cmap="viridis"
        )
        plt.colorbar(sc, ax=ax, label="Alpha (α)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Drift Trajectory of Equilibrium")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="box")
