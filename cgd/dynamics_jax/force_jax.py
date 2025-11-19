# cgd/dynamics_jax/force_jax.py
"""
cgd/dynamics_jax/force_jax.py

通过JAX的自动微分功能，实现合力 F_net 的精确、高性能计算。
"""

import jax.numpy as jnp
from jax import grad, jit
from functools import partial

# --- 核心依赖：从本模块的 potential_jax 导入 ---
from cgd.dynamics_jax.potential_jax import potential_gaussian_jax

# --- 1. 使用 jax.grad 自动推导梯度函数 ---
# 我们告诉JAX，我们想要的是 potential_gaussian_jax 相对于其第一个参数(p)的梯度。
# 这个 grad_potential_fn 本身就是一个新的、可被调用的函数！
grad_potential_fn = grad(potential_gaussian_jax, argnums=0)

# --- 2. 编译最终的力计算函数 ---

@partial(jit, static_argnames=("K",))
def calculate_F_net_jax(
    p: jnp.ndarray, 
    sources_p_stim: jnp.ndarray, 
    sources_strength: jnp.ndarray, 
    alpha: float, 
    radius: float,
    K: int
) -> jnp.ndarray:
    """
    通过自动微分计算在点p的合力 F_net (势能的负梯度) (JAX版本)。

    Args:
        (参数与 potential_gaussian_jax 完全相同)

    Returns:
        jnp.ndarray: 在点p的合力向量 F_net，它是一个切向量（和为0）。
    """
    # 直接调用JAX为我们自动生成的梯度函数
    grad_vec = grad_potential_fn(p, sources_p_stim, sources_strength, alpha, radius, K)
    
    # 力是势能的负梯度，并需要投影到切空间（和为零）
    force_net = -(grad_vec - jnp.mean(grad_vec))
    
    return force_net