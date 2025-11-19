# cgd/measure/calibration.py

from typing import List, Dict, Any, Tuple, Optional, Literal
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import os
import warnings

# 从我们的其他模块中导入所需的对象和工具
from ..core.source import GravitationalSource
from ..core.universe import Universe
from ..geometry.simplex import distance_FR

class AlphaFitter:
    """
    一个高性能的、用于从实验数据中全局拟合最优 alpha 值的类。
    支持 'numpy' 和 'jax' 双计算后端。
    """
    def __init__(self, 
                 source_map: Dict[str, GravitationalSource],
                 competition_data: Dict[str, Dict[str, Any]],
                 backend: Literal['numpy', 'jax'] = 'numpy'):
        """
        初始化 AlphaFitter。
        ...
        """
        self.source_map = source_map
        self.competition_data = competition_data
        self.backend = backend
        
        # JAX 可用性检查
        if self.backend == 'jax':
            try:
                import jax
            except ImportError:
                raise ImportError("无法加载 JAX 后端。请确保已安装 JAX。")

        # --- 新增：坐标系一致性检查 ---
        # 筛选出所有 'substitution' 类型的源
        substitution_sources = [
            source for source in self.source_map.values() 
            if source.v_type == 'substitution'
        ]

        # 如果存在多个 substitution 源，则必须检查它们的一致性
        if len(substitution_sources) > 1:
            first_sub_labels = substitution_sources[0].labels
            
            for source in substitution_sources[1:]:
                if source.labels != first_sub_labels:
                    raise ValueError(
                        f"AlphaFitter 初始化失败：坐标系不一致。\n"
                        f"检测到多个 'substitution' 类型的引力源，但它们的测量坐标系 (labels) 不同。\n"
                        f" - 源 '{substitution_sources[0].name}' 的坐标系: {first_sub_labels}\n"
                        f" - 源 '{source.name}' 的坐标系: {source.labels}\n"
                        f"为了进行有效的全局 alpha 拟合，所有 'substitution' 类型的源必须在"
                        f"同一个参照系下测量（即拥有完全相同的 labels）。"
                    )
        # --- 检查结束 ---

        self._precompute_sources()

    def _precompute_sources(self):
        """预计算每个实验所需的、经过维度嵌入的引力源列表。"""
        self.exp_sources = {}
        for exp_name, exp_details in self.competition_data.items():
            new_labels = exp_details.get('labels')
            source_names = exp_details.get('sources')

            if new_labels is None or source_names is None:
                raise ValueError(f"实验 '{exp_name}' 的数据缺少 'labels' 或 'sources' 键。")

            sources_for_exp = [self.source_map[name].embed(list(new_labels)) for name in source_names]
            self.exp_sources[exp_name] = sources_for_exp

    def _objective_function_single(self, alpha_candidate: float, solver_options: dict) -> float:
        """为单个 alpha 候选值计算总误差。"""
        if alpha_candidate < 0: return np.inf

        total_error = 0.0
        for exp_name, exp_details in self.competition_data.items():
            K = exp_details['K']
            p_observed = exp_details['p_observed']
            labels = exp_details['labels']
            
            # --- 核心升级：创建携带 backend 信息的 Universe ---
            universe = Universe(K=K, alpha=alpha_candidate, labels=labels, backend=self.backend)
            
            sources_for_exp = self.exp_sources[exp_name]

            # Universe 会自动调用正确后端的求解器
            equilibria = universe.find_equilibria(sources_for_exp, **solver_options)

            if not equilibria: return np.inf

            # 选择与观测值最近的解来计算误差
            distances = [distance_FR(p, p_observed) for p in equilibria]
            min_distance = np.min(distances)
            total_error += min_distance**2

        return total_error

    def _objective_function_for_minimize(self, alpha_array: np.ndarray, solver_options: dict) -> float:
        """scipy.minimize 使用的包装函数。"""
        return self._objective_function_single(alpha_array[0], solver_options)

    def fit(self, 
            alpha_range: Tuple[float, float] = (0.01, 5.0), 
            grid_points: int = 20, # 默认值可以降低，因为JAX很快
            n_jobs: int = -1,
            solver_options: Optional[Dict[str, Any]] = None) -> float:
        """
        执行一个稳健的、并行的全局拟合，找到最优的 alpha 值。
        """
        if n_jobs == -1: n_jobs = os.cpu_count() or 1
        
        # --- 智能准备 solver_options ---
        if solver_options is None:
            if self.backend == 'jax':
                # JAX 后端可以接受更多超参数
                solver_options = {
                    'num_random_seeds': 10, 'validate_stability': False,
                    'max_rounds': 50, 'cd_steps': 15, 'learning_rate': 0.05
                }
            else: # numpy
                solver_options = {'num_random_seeds': 10, 'validate_stability': False}
        
        # 确保并行任务不会再次并行
        solver_options['n_jobs'] = 1

        print(f"AlphaFitter (backend='{self.backend}'): "
              f"正在使用 {n_jobs} 个核心进行并行网格搜索 ({grid_points}个点)...")

        # --- 网格搜索与局部优化 (逻辑保持不变) ---
        alpha_candidates = np.linspace(alpha_range[0], alpha_range[1], grid_points)
        errors = Parallel(n_jobs=n_jobs)(
            delayed(self._objective_function_single)(alpha, solver_options) for alpha in alpha_candidates
        )
        # ... (后续的错误检查、寻找最优初值、minimize调用等逻辑完全不变) ...
        errors = np.array(errors)
        valid_indices = np.where(np.isfinite(errors))[0]
        if len(valid_indices) == 0:
            raise RuntimeError("Alpha 拟合失败：所有网格点的误差均为无穷大。")

        best_grid_index = valid_indices[np.argmin(errors[valid_indices])]
        alpha_optimal_coarse = alpha_candidates[best_grid_index]
        print(f"网格搜索找到的最佳 alpha 初步估计: {alpha_optimal_coarse:.4f}")

        print("AlphaFitter: 正在进行局部的精确优化...")
        search_width = (alpha_range[1] - alpha_range[0]) / (grid_points - 1) * 2 if grid_points > 1 else 0.5
        fine_tune_bounds = (max(0.0, alpha_optimal_coarse - search_width), alpha_optimal_coarse + search_width)

        result = minimize(
            self._objective_function_for_minimize,
            x0=[alpha_optimal_coarse],
            args=(solver_options,),
            bounds=[fine_tune_bounds],
            method='L-BFGS-B',
            options={'ftol': 1e-12, 'gtol': 1e-9}
        )

        if result.success:
            print("局部精确优化成功。")
            return result.x[0]
        else:
            warnings.warn(f"局部精确优化未能收敛 ({result.message})，将返回网格搜索的结果。")
            return alpha_optimal_coarse