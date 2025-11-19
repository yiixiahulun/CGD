"""
cgd.core

定义CGD框架的核心数据结构。
"""

from .source import GravitationalSource
from .universe import Universe

__all__ = ["GravitationalSource", "Universe"]
