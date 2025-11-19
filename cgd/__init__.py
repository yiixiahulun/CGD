# cgd/__init__.py
"""
Cognitive Gravitational Dynamics (cgd)

一个用于建模与预测概率系统的第一性原理物理框架的Python实现。
"""
# 定义版本号，反映重大的API升级和理论完善
__version__ = "3.0.0"

# --- 从子模块中提升最核心的、用户最常用的对象和函数 ---

# 1. 核心对象 (Core Objects) - 用户构建CGD世界的基本积木
from .core import GravitationalSource, Universe

# 2. 核心测量与发现工具 (Measurement & Discovery Tools)
#    这些是连接实验数据和CGD理论实体的核心算法接口。
from .measure import AlphaFitter, ProbeFitter, VectorInverter, measure_source

# 3. 高级工作流 (High-Level Workflows)
#    为标准分析任务提供端到端的解决方案，是新用户的最佳入口。
from .workflows import run_measurement_pipeline, run_universe_analysis

# 定义顶层包的公开API，并附上注释以解释其角色
__all__ = [
    # === 核心对象 (Nouns) ===
    # 用于定义宇宙的物理学和物质的基本单元。
    "Universe",
    "GravitationalSource",
    # === 核心测量与发现 (Verbs) ===
    # 用于从实验数据创建理论实体，或从偏差中发现新理论。
    "measure_source",  # 主要的测量函数，从数据创建带类型的 Source 对象。
    "AlphaFitter",  # 用于校准宇宙的状态参数 alpha。
    "ProbeFitter",  # 用于从预测偏差中发现新的效应（探针）。
    "VectorInverter",  # (高级) 用于“靴袢式”流程中的v_eigen精炼。
    # === 高级工作流 (Recipes) ===
    # 封装了标准科学研究流程的端到端函数。
    "run_measurement_pipeline",  # 从原始数据到 source_map 和 alpha 的一站式测量流程。
    "run_universe_analysis",  # 使用已测量的源，对一个宇宙进行完整的分析、预测和诊断。
]

# --- 备注：被有意降级的接口 ---
# 以下函数是更底层的实现细节，或已被更高级的函数封装。
# 我们不将它们暴露在顶层命名空间，以保持API的简洁和引导性。
# 高级用户如果确实需要，仍可以通过 `from cgd.utils import ...` 或
# `from cgd.geometry import ...` 的方式访问它们。
