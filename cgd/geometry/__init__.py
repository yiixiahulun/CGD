# cgd/geometry/__init__.py (最终版)

"""
cgd.geometry

提供与信息几何相关的纯数学函数和类。
这个模块同时提供了NumPy和JAX的实现。
"""
# --- NumPy 实现 (默认) ---
from .simplex import Simplex, distance_FR, get_chaos_origin, get_radius
from .transformations import log_map_from_origin, exp_map_from_origin

# --- JAX 实现 (尝试导入) ---
# 将JAX的实现也放在这里，作为这个模块的一部分
try:
    from ._jax_impl import (
        get_chaos_origin_jax, 
        distance_FR_jax,
        log_map_from_origin_jax,
        exp_map_from_origin_jax
    )
    JAX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # 如果JAX未安装或_jax_impl文件不存在，优雅地处理
    JAX_AVAILABLE = False
    # 定义“伪函数”，以便类型检查通过，并在运行时给出明确错误
    def _jax_not_found(*args, **kwargs):
        raise ImportError("JAX backend is not available. Please install JAX.")
    get_chaos_origin_jax = distance_FR_jax = log_map_from_origin_jax = exp_map_from_origin_jax = _jax_not_found

# 定义这个模块的公共API
__all__ = [
    # NumPy
    'Simplex', 'distance_FR', 'get_chaos_origin', 'get_radius',
    'log_map_from_origin', 'exp_map_from_origin',
    # JAX
    'get_chaos_origin_jax', 'distance_FR_jax',
    'log_map_from_origin_jax', 'exp_map_from_origin_jax',
    # Flag
    'JAX_AVAILABLE'
]