from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
import numpy as np
from typing import List, Dict, TYPE_CHECKING
from itertools import combinations
from joblib import Parallel, delayed
import os
import warnings
from scipy.stats import qmc

# --- 核心依赖 ---
if TYPE_CHECKING:
    from ..core import Universe, GravitationalSource

# --- 导入正确的函数！ ---
# JAX 版本的几何工具
from cgd.core import Universe, GravitationalSource
from ..geometry import exp_map_from_origin_jax, exp_map_from_origin
from .potential_jax import potential_gaussian_jax
from .force_jax import calculate_F_net_jax
from ..geometry import exp_map_from_origin 



# 为了独立，我们先在这里定义
def exp_map_from_origin(v):
    v=np.asarray(v,dtype=float); K=len(v); o=np.full(K,1.0/K); u=v*np.sqrt(o); n_u=np.linalg.norm(u)
    if n_u<1e-12: return o
    s_p=np.cos(n_u/2)*np.sqrt(o)+np.sin(n_u/2)*(u/n_u); p=s_p**2; return p/np.sum(p)


# ============================================================================
# === JAX化的核心求解器函数 ================================================
# ============================================================================
@partial(jit, static_argnames=("K", "max_rounds", "cd_steps", "learning_rate"))
def _solve_cd_single_jax(
    p_initial: jnp.ndarray, 
    sources_p_stim: jnp.ndarray, 
    sources_strength: jnp.ndarray, 
    alpha: float, 
    radius: float,
    K: int,
    max_rounds: int,
    cd_steps: int,
    learning_rate: float
) -> jnp.ndarray:
    """一个完全JAX化的坐标下降求解器，可被JIT编译。"""
    
    def find_min_on_slice_jax(p_start, d_opt):
        d_comp = K - 1
        s_val = 1.0 - jnp.sum(p_start) + p_start[d_opt] + p_start[d_comp]
        s_val = jnp.maximum(s_val, 1e-9)

        def objective_1d(x):
            p_vec = p_start.at[d_opt].set(x).at[d_comp].set(s_val - x)
            return potential_gaussian_jax(p_vec, sources_p_stim, sources_strength, alpha, radius, K)
        
        grad_1d = grad(objective_1d)
        
        def gd_step(_, x_current):
            g = grad_1d(x_current)
            x_next = x_current - learning_rate * g
            return jnp.clip(x_next, 0, s_val)

        final_x = jax.lax.fori_loop(0, cd_steps, gd_step, s_val / 2.0)
        p_res = p_start.at[d_opt].set(final_x).at[d_comp].set(s_val - final_x)
        return p_res

    def main_loop_body(_, p_current):
        def cd_loop_body(i, p_inner):
            return find_min_on_slice_jax(p_inner, i)
        
        p_after_round = jax.lax.fori_loop(0, K - 1, cd_loop_body, p_current)
        return p_after_round
        
    p_final = jax.lax.fori_loop(0, max_rounds, main_loop_body, p_initial)
    return p_final / jnp.sum(p_final)

# ============================================================================
# === EquilibriumFinderJax 类 ================================================
# ============================================================================

class EquilibriumFinderJax:
    """EquilibriumFinder 的高性能 JAX 版本。"""
    def __init__(self, universe: 'Universe', sources: List['GravitationalSource']):
        self.universe = universe
        self.sources = sources
        self.K = universe.K
        self._prepare_jax_inputs()

    def _prepare_jax_inputs(self):
        """将对象列表扁平化为JAX可以处理的数组。"""
        # --- 关键修复：这里必须使用NumPy版本，因为v_eigen是NumPy数组 ---
        p_stims_np = [exp_map_from_origin(s.v_eigen) for s in self.sources]
        strengths_np = [s.strength for s in self.sources]
        
        self.sources_p_stim_jax = jnp.array(p_stims_np)
        self.sources_strength_jax = jnp.array(strengths_np)

    def _generate_intelligent_guesses(self, num_random_seeds: int) -> List[np.ndarray]:
        """生成一系列高质量的初始猜测点。"""
        k = self.K
        guesses = [self.universe.simplex.origin] # 总是包含宇宙中心

        # LESA 预测点
        try:
            v_net_lesa = sum(s.embed(list(self.universe.labels)).v_eigen for s in self.sources)
            p_lesa_guess = exp_map_from_origin(v_net_lesa)
            guesses.append(p_lesa_guess)
        except Exception as e:
            warnings.warn(f"LESA 初始点计算失败: {e}")
        
        # 顶点和边界点
        for i in range(k):
            p = np.full(k, 0.01 / (k - 1) if k > 1 else 0)
            p[i] = 0.99
            guesses.append(p)
        if k > 2:
            for i, j in combinations(range(k), 2):
                p = np.zeros(k); p[i], p[j] = 0.5, 0.5
                guesses.append(p)
                
        # 准随机采样点
        if num_random_seeds > 0:
            n_power_of_2 = 1 << (num_random_seeds - 1).bit_length()
            try:
                sampler = qmc.Sobol(d=k, scramble=True)
                samples = sampler.random(n=n_power_of_2)
                samples /= np.sum(samples, axis=1, keepdims=True)
                guesses.extend(samples)
            except Exception:
                guesses.extend(np.random.dirichlet(np.ones(k), size=num_random_seeds))
                
        return guesses
        
    def _solve_from_single_point_wrapper(self, p_initial_np: np.ndarray, solver_params: Dict) -> np.ndarray:
        """Python包装器，调用JIT编译的JAX核心求解器。"""
        p_initial_jax = jnp.array(p_initial_np)
        
        p_final_jax = _solve_cd_single_jax(
            p_initial=p_initial_jax,
            sources_p_stim=self.sources_p_stim_jax,
            sources_strength=self.sources_strength_jax,
            alpha=self.universe.alpha,
            radius=self.universe.simplex.radius,
            K=self.K,
            **solver_params
        )
        return np.array(p_final_jax)
        
    def _validate_point_stability_jax(self, p_star: np.ndarray, grad_tolerance: float = 1e-5) -> bool:
        """使用JAX后端进行稳定性验证 (仅检查梯度)。"""
        p_star_jax = jnp.array(p_star)
        force_jax = calculate_F_net_jax(p_star_jax, self.sources_p_stim_jax, self.sources_strength_jax, self.universe.alpha, self.universe.simplex.radius, self.K)
        return np.linalg.norm(force_jax) < grad_tolerance

    def find(self, 
             num_random_seeds: int = 20, 
             uniqueness_tolerance: float = 1e-4, 
             validate_stability: bool = False, 
             n_jobs: int = -1,
             max_rounds: int = 50,
             cd_steps: int = 15,
             learning_rate: float = 0.05
             ) -> List[np.ndarray]:
        
        if n_jobs == -1: n_jobs = os.cpu_count() or 1
            
        initial_guesses = self._generate_intelligent_guesses(num_random_seeds)
        
        solver_params = {
            'max_rounds': max_rounds,
            'cd_steps': cd_steps,
            'learning_rate': learning_rate
        }
        
        candidate_points = Parallel(n_jobs=n_jobs)(
            delayed(self._solve_from_single_point_wrapper)(guess, solver_params) for guess in initial_guesses
        )
        
        unique_candidates = []
        if candidate_points:
            # 使用更高效和健壮的去重逻辑
            for p in candidate_points:
                if p is not None and np.isfinite(p).all():
                    is_close_to_existing = False
                    for up in unique_candidates:
                        if distance_FR(p, up) < uniqueness_tolerance:
                            is_close_to_existing = True
                            break
                    if not is_close_to_existing:
                        unique_candidates.append(p)

        if not validate_stability:
            return sorted(unique_candidates, key=lambda p: -p.max())
        else:
            print("正在使用JAX后端进行稳定性验证 (仅检查梯度)...")
            final_solutions = [p for p in unique_candidates if self._validate_point_stability_jax(p)]
            return sorted(final_solutions, key=lambda p: -p.max())