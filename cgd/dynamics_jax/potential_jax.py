# cgd/dynamics_jax/potential_jax.py (修复版)

"""
cgd.dynamics_jax/potential_jax.py

提供CGD势能计算的JAX实现。
"""
import jax.numpy as jnp
from jax import jit
from functools import partial

# --- 核心依赖：直接从JAX的几何实现导入 ---
from ..geometry import distance_FR_jax

@partial(jit, static_argnames=("K",))
def potential_gaussian_jax(
    p: jnp.ndarray, 
    sources_p_stim: jnp.ndarray, 
    sources_strength: jnp.ndarray, 
    alpha: float, 
    radius: float,
    K: int
) -> float:
    """
    计算总高斯势能 (JAX版本)。
    """
    radius = jnp.maximum(radius, 1e-12)

    def body_fun(i, current_potential):
        p_stim = sources_p_stim[i]
        strength = sources_strength[i]
        
        # --- 确保调用的是JAX版的距离函数 ---
        relative_dist = distance_FR_jax(p, p_stim) / radius
        
        potential_s = -strength * jnp.exp(-((alpha * relative_dist)**2))
        return current_potential + potential_s

    total_potential = jax.lax.fori_loop(0, len(sources_strength), body_fun, 0.0)
    return total_potential