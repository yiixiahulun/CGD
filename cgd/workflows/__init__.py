# cgd/workflows/__init__.py
"""
cgd.workflows

将底层模块组合成面向具体科学问题的高级函数。
"""

# --- 核心修改：使用新的、重构后的函数名 ---
from .analysis import run_measurement_pipeline, run_universe_analysis

# 定义这个 workflows 包的公共API
__all__ = [
    'run_measurement_pipeline',  # <--- 使用新名字
    'run_universe_analysis'      # <--- 使用新名字
]