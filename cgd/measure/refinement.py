# cgd/measure/refinement.py
"""
实现一个高性能的 VectorInverter 类，用于“靴袢式解耦”流程中的“粒子精炼”步骤。
"""

import numpy as np
from scipy.optimize import minimize
from typing import List

# --- 从我们库的其他部分导入必要的对象和函数 ---
from ..core.source import GravitationalSource
from ..core.universe import Universe
from ..dynamics.force import calculate_F_net

class VectorInverter:
    """
    一个使用FDS前向模型，从观测数据中高效反演 v_eigen 的优化器。
    
    它通过调整候选v_eigen，使得由其和舞台源共同作用下，在目标观测点
    p_total_obs 产生的合力为零，从而反演出在给定alpha宇宙中的精确v_eigen。
    这是“粒子精炼”的核心工具。
    """
    def __init__(self, 
                 p_total_obs: np.ndarray, 
                 stage_source: GravitationalSource,
                 universe: Universe,
                 target_source_name: str,
                 target_source_type: str):
        """
        初始化 VectorInverter。

        Args:
            p_total_obs (np.ndarray): 
                单源净化实验的目标观测概率。
            stage_source (GravitationalSource): 
                一个代表了舞台效应的、完整的引力源对象。
            universe (Universe): 
                一个已用固定的 alpha_est 初始化的宇宙。
            target_source_name (str):
                我们正在精炼的目标引力源的名称。
            target_source_type (str):
                我们正在精炼的目标引力源的 v_type ('absolute' 或 'substitution')。
        """
        self.p_total_obs = p_total_obs
        self.universe = universe
        self.stage_source = stage_source
        self.target_source_name = target_source_name
        self.target_source_type = target_source_type
        self.K = universe.K
        
        # 从舞台源继承标签，因为它们必须在同一个坐标系下
        self.labels = stage_source.labels 

    def _objective_function_fast(self, v_eigen_candidate_flat: np.ndarray) -> float:
        """
        快速目标函数：计算在目标观测点 p_total_obs 的合力 F_net 的模长平方。
        """
        v_eigen_candidate = v_eigen_candidate_flat - np.mean(v_eigen_candidate_flat)
        
        # 创建候选的引力源，必须提供完整的类型和标签信息
        candidate_source = GravitationalSource(
            name=f"Candidate_{self.target_source_name}", 
            v_eigen=v_eigen_candidate,
            v_type=self.target_source_type,
            labels=self.labels
        )
        
        sources = [self.stage_source, candidate_source]
        force_at_target = calculate_F_net(self.p_total_obs, sources, self.universe)
        
        return np.sum(force_at_target**2)

    def refine(self, initial_guess_v_eigen: np.ndarray) -> GravitationalSource:
        """
        执行高效的精炼优化，并返回一个完整的 GravitationalSource 对象。

        Args:
            initial_guess_v_eigen (np.ndarray): v_eigen 的初始猜测值 (通常是零阶估计)。

        Returns:
            GravitationalSource: 一个包含精炼后 v_eigen 的、信息完备的引力源对象。
        """
        print(f"  - 正在为 '{self.target_source_name}' (strength≈{np.linalg.norm(initial_guess_v_eigen):.3f}) 进行精炼...")

        initial_guess_flat = initial_guess_v_eigen
        
        result = minimize(
            self._objective_function_fast,
            x0=initial_guess_flat,
            method='BFGS',
            options={'gtol': 1e-9, 'maxiter': 1000}
        )
        
        v_refined_flat = result.x
        v_refined = v_refined_flat - np.mean(v_refined_flat)

        if result.success and result.fun < 1e-9:
            print(f"  - 精炼成功！新 strength={np.linalg.norm(v_refined):.3f} (误差: {result.fun:.2e})")
        else:
            print(
                f"  - 警告: VectorInverter 未能收敛或未达精度要求。"
                f"Message: {result.message}, Final Error: {result.fun:.2e}"
            )
        
        # 无论是否完美收敛，都构建并返回一个完整的 Source 对象
        return GravitationalSource(
            name=self.target_source_name,
            v_eigen=v_refined,
            v_type=self.target_source_type,
            labels=self.labels
        )