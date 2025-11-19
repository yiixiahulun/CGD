# cgd/dynamics/solvers.py
"""
cgd.dynamics.solvers

实现CGD动力学的核心求解器：
- EquilibriumFinder: 一个高性能的、用于寻找所有稳定吸引子的全局求解器。
- TrajectorySimulator: 一个用于模拟系统演化轨迹的常微分方程求解器。
"""
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import qmc
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from itertools import combinations
from joblib import Parallel, delayed
import os
import warnings

# --- 导入我们的核心组件 ---
if TYPE_CHECKING:
    from ..core.universe import Universe
    from ..core.source import GravitationalSource

from ..geometry import distance_FR, exp_map_from_origin
from ..dynamics import potential_gaussian, calculate_F_net

# (辅助函数保持不变)

class EquilibriumFinder:
    """
    一个高性能的、用于在给定CGD宇宙中寻找所有稳定平衡点（吸引子）的类。
    """
    def __init__(self, universe: 'Universe', sources: List['GravitationalSource']):
        self.universe = universe
        self.sources = sources
        self.K = universe.K
    
    # ... (_find_min_on_slice 和 _solve_cd_from_single_point 保持不变) ...
    def _find_min_on_slice(self, p_start, dim_to_optimize, compensating_dim):
        k = self.K; fixed_indices = [i for i in range(k) if i != dim_to_optimize and i != compensating_dim]
        fixed_sum = np.sum(p_start[fixed_indices]); sum_val = 1.0 - fixed_sum
        if sum_val < 0: return p_start
        def objective_1d(x):
            p_vec = np.zeros(k); p_vec[dim_to_optimize] = x; p_vec[compensating_dim] = sum_val - x
            if fixed_indices: p_vec[fixed_indices] = p_start[fixed_indices]
            return potential_gaussian(p_vec, self.sources, self.universe)
        res = minimize_scalar(objective_1d, bounds=(0, sum_val), method='bounded', options={'xatol': 1e-10})
        p_res = np.zeros(k); p_res[dim_to_optimize] = res.x; p_res[compensating_dim] = sum_val - res.x
        if fixed_indices: p_res[fixed_indices] = p_start[fixed_indices]
        return p_res

    def _solve_cd_from_single_point(self, p_initial, tolerance=1e-8, max_rounds=50):
        p_current = np.array(p_initial, dtype=float)
        if np.abs(np.sum(p_current) - 1.0) > 1e-9: p_current /= np.sum(p_current)
        compensating_dim = self.K - 1; dims_to_optimize = list(range(self.K - 1))
        for _ in range(max_rounds):
            p_before_round = p_current.copy()
            for i in dims_to_optimize: p_current = self._find_min_on_slice(p_current, i, compensating_dim)
            if np.abs(np.sum(p_current) - 1.0) > 1e-9: p_current /= np.sum(p_current)
            if distance_FR(p_current, p_before_round) < tolerance: break
        return p_current

    def _generate_intelligent_guesses(self, num_random_seeds: int) -> List[np.ndarray]:
        """
        生成一系列高质量的初始猜测点，用于全局优化。
        """
        k = self.K
        guesses = []
        
        # --- LESA 预测点 (最终修复版) ---
        try:
            v_net_lesa = np.zeros(k)
            # 从 universe 对象获取当前宇宙的坐标系定义
            universe_labels = list(self.universe.labels)

            for s in self.sources:
                # 直接调用 source 对象的智能 embed 方法
                embedded_source = s.embed(new_labels=universe_labels)
                v_net_lesa += embedded_source.v_eigen
            
            p_lesa_guess = exp_map_from_origin(v_net_lesa)
            guesses.extend([self.universe.simplex.origin, p_lesa_guess])
        except Exception as e:
            warnings.warn(
                f"LESA 初始点计算失败 ({type(e).__name__}: {e})。"
                "这可能是由于源标签不兼容导致的。将只使用宇宙中心点作为启发式起点。"
            )
            guesses.append(self.universe.simplex.origin)
            
        # --- 其他确定性猜测点 ---
        for i in range(k):
            p = np.full(k, 0.01 / (k - 1) if k > 1 else 0)
            p[i] = 0.99
            guesses.append(p)
            
        if k > 2:
            for i, j in combinations(range(k), 2):
                p = np.zeros(k)
                p[i], p[j] = 0.5, 0.5
                guesses.append(p)
                
        # --- 准随机采样点 ---
        if num_random_seeds > 0:
            # 将种子数调整为最接近的2的幂，以获得更好的Sobol序列性能
            n_power_of_2 = 1 << (num_random_seeds - 1).bit_length() if num_random_seeds > 0 else 0
            if n_power_of_2 > 0:
                try:
                    sampler = qmc.Sobol(d=k, scramble=True)
                    samples = sampler.random(n=n_power_of_2)
                    samples /= np.sum(samples, axis=1, keepdims=True)
                    guesses.extend(samples)
                except Exception:
                    # 如果Sobol失败，回退到Dirichlet分布
                    guesses.extend(np.random.dirichlet(np.ones(k), size=num_random_seeds))
                
        return guesses

    # ... (后续代码 _calculate_hessian, _validate_point_stability, find 保持不变) ...
    def _calculate_hessian(self, p_star: np.ndarray, h: float = 1e-5) -> np.ndarray:
        hessian = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i, self.K):
                p_ii, p_io, p_oi, p_oo = p_star.copy(), p_star.copy(), p_star.copy(), p_star.copy()
                p_ii[i] += h; p_ii[j] += h; p_io[i] += h; p_io[j] -= h
                p_oi[i] -= h; p_oi[j] += h; p_oo[i] -= h; p_oo[j] -= h
                for p_vec in [p_ii, p_io, p_oi, p_oo]: p_vec /= np.sum(p_vec)
                U_ii = potential_gaussian(p_vec, self.sources, self.universe)
                U_io = potential_gaussian(p_vec, self.sources, self.universe)
                U_oi = potential_gaussian(p_vec, self.sources, self.universe)
                U_oo = potential_gaussian(p_vec, self.sources, self.universe)
                hessian[i, j] = hessian[j, i] = (U_ii - U_io - U_oi + U_oo) / (4 * h**2)
        return hessian

    def _validate_point_stability(self, p_star: np.ndarray, grad_tolerance: float = 1e-4) -> str:
        force = calculate_F_net(p_star, self.sources, self.universe)
        if np.linalg.norm(force) > grad_tolerance:
            return f"Unstable (High Gradient: {np.linalg.norm(force):.2e})"
        hessian = self._calculate_hessian(p_star)
        eigenvalues = eigh(hessian, eigvals_only=True)
        relevant_eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-6]
        if np.any(relevant_eigenvalues < 0):
            num_negative = np.sum(relevant_eigenvalues < 0)
            return f"Unstable (Saddle Point, {num_negative} negative eigval(s))"
        return "Stable"

    def find(self, num_random_seeds=20, uniqueness_tolerance=1e-4, validate_stability=False, n_jobs=-1):
        if n_jobs == -1: n_jobs = os.cpu_count() or 1
        initial_guesses = self._generate_intelligent_guesses(num_random_seeds)
        candidate_points = Parallel(n_jobs=n_jobs)(delayed(self._solve_cd_from_single_point)(guess) for guess in initial_guesses)
        unique_candidates = []
        if candidate_points:
            for p in candidate_points:
                if p is not None and np.isfinite(p).all():
                    if not any(np.allclose(p, up, atol=uniqueness_tolerance) for up in unique_candidates):
                        unique_candidates.append(p)
        if not validate_stability:
            return sorted(unique_candidates, key=lambda p: p[0], reverse=True)
        else:
            final_solutions = []
            for p_candidate in unique_candidates:
                verdict = self._validate_point_stability(p_candidate)
                if verdict == "Stable": final_solutions.append(p_candidate)
            return sorted(final_solutions, key=lambda p: p[0], reverse=True)


class TrajectorySimulator:
    # ... (保持不变) ...
    def __init__(self, universe, sources, p_initial, T, num_points):
        self.universe=universe; self.sources=sources; self.p_initial=p_initial; self.T=T; self.num_points=num_points
    def _dynamics(self, t, p):
        p_normalized = np.maximum(p, 0); p_normalized /= np.sum(p_normalized)
        return calculate_F_net(p_normalized, self.sources, self.universe)
    def simulate(self):
        t_eval = np.linspace(0, self.T, self.num_points)
        solution = solve_ivp(self._dynamics, [0, self.T], self.p_initial, t_eval=t_eval, method='RK45', dense_output=True)
        trajectory = solution.y.T; trajectory_normalized = np.array([row / np.sum(row) for row in trajectory])
        return solution.t, trajectory_normalized