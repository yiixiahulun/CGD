import jax
import jax.numpy as jnp
from jax import jit, partial

EPSILON = 1e-12

# --- 核心修复：告诉JIT编译器，K是一个静态参数 ---
@partial(jit, static_argnames=('K',))
def get_chaos_origin_jax(K: int) -> jnp.ndarray:
    return jnp.full(K, 1.0 / K)

@jit
def distance_FR_jax(p: jnp.ndarray, q: jnp.ndarray) -> float:
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)
    p_safe = jnp.maximum(p, 0)
    q_safe = jnp.maximum(q, 0)
    dot_product = jnp.clip(jnp.sum(jnp.sqrt(p_safe * q_safe)), -1.0, 1.0)
    return 2 * jnp.arccos(dot_product)

@jit
def log_map_from_origin_jax(p: jnp.ndarray) -> jnp.ndarray:
    # --- 核心修复：在函数内部推断K，而不是作为参数传入 ---
    K = p.shape[0] 
    # JAX现在可以从具体数组p中推断出K的值，所以get_chaos_origin_jax可以工作
    origin = get_chaos_origin_jax(K)
    p_normalized = p / jnp.sum(p)
    dot_product = jnp.clip(jnp.sum(jnp.sqrt(origin * p_normalized)), -1.0, 1.0)
    theta = jnp.arccos(dot_product)
    
    def compute_v(_):
        u_raw = jnp.sqrt(p_normalized) - dot_product * jnp.sqrt(origin)
        norm_u = jnp.linalg.norm(u_raw)
        def u_is_nonzero(_):
            v_raw = (2 * theta / norm_u) * u_raw / jnp.sqrt(origin)
            return v_raw - jnp.mean(v_raw)
        def u_is_zero(_):
            return jnp.zeros_like(p)
        return jax.lax.cond(norm_u < EPSILON, u_is_zero, u_is_nonzero, operand=None)

    def return_zero_vector(_):
        return jnp.zeros_like(p)

    return jax.lax.cond(theta < EPSILON, return_zero_vector, compute_v, operand=None)


@jit
def exp_map_from_origin_jax(v: jnp.ndarray) -> jnp.ndarray:
    # --- 核心修复：同上，在内部推断K ---
    K = v.shape[0]
    origin = get_chaos_origin_jax(K)
    v_centered = v - jnp.mean(v)
    u = v_centered * jnp.sqrt(origin)
    norm_u = jnp.linalg.norm(u)
    
    def v_is_nonzero(_): 
        sqrt_p = jnp.cos(norm_u / 2.0) * jnp.sqrt(origin) + jnp.sin(norm_u / 2.0) * (u / norm_u)
        p = sqrt_p**2
        return p / jnp.sum(p)

    def v_is_zero(_): 
        return origin
        
    return jax.lax.cond(norm_u < EPSILON, v_is_zero, v_is_nonzero, operand=None)