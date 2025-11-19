# cgd/measure/__init__.py

# --- 核心修改：将 measure_source 作为主要测量接口 ---
# 我们仍然可以从 purification.py 导入，即使它包含了旧的别名
from .calibration import AlphaFitter
from .discovery import ProbeFitter
from .purification import geometric_purification, measure_source
from .refinement import VectorInverter

__all__ = [
    # --- 核心测量工具 ---
    "measure_source",  # <--- 新的、推荐的测量函数
    # --- 核心算法类 ---
    "AlphaFitter",
    "ProbeFitter",
    "VectorInverter",
    # --- (可选) 为需要访问底层数学的用户保留 ---
    # 或者为了保持API整洁，甚至可以考虑不暴露它
    "geometric_purification",
]
