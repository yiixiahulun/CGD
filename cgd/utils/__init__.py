# cgd/utils/__init__.py
"""
cgd.utils

提供一系列辅助函数，用于简化从不同数据格式到CGD核心对象的转换。
"""

# 从 data_helpers.py 文件中导入我们希望公开给外部的函数
from .data_helpers import create_source_map_from_data, process_counts_to_p_dict

# 定义这个 utils 包的公共API
__all__ = [
    "process_counts_to_p_dict",  # 底层转换工具
    "create_source_map_from_data",  # 高级、推荐的一站式工具
]
