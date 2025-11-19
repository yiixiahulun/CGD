# cgd/core/source.py
"""
cgd.core.source

这个模块定义了 GravitationalSource 类。
这个类封装了一个引力源的内在属性，并“知道”自己的理论层级
('absolute' vs 'substitution') 和原始坐标系 ('labels')，
以确保正确的科学使用和无歧义的维度变换。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Tuple
import numpy as np
import warnings

# --- 核心依赖：为了 embed 方法，我们需要导入核心几何工具 ---
from ..geometry.transformations import exp_map_from_origin, log_map_from_origin

@dataclass(frozen=True)
class GravitationalSource:
    """
    一个封装了引力源内在属性、理论类型和坐标系标签的不可变数据类。

    这是CGD宇宙中“物质”的数字化表示。每个源都由其“物理基因”(v_eigen)，
    其所属的理论层级(v_type)，以及其被测量时所处的坐标系(labels)来完整定义。

    Attributes:
        name (str): 
            引力源的唯一名称。
        v_eigen (np.ndarray): 
            其本征效应向量，一个和为零的numpy数组。
        v_type (Literal['absolute', 'substitution']):
            v_eigen的理论类型，'absolute'拥有跨维度能力，'substitution'仅限同维度。
        labels (Tuple[str, ...]):
            一个包含原子事件标签的元组，定义了 v_eigen 各分量对应的坐标轴。
            这是实现无歧义维度变换的基石。
        strength (float): 
            v_eigen的欧几里得模长 (||v_eigen||)，代表其绝对内在强度。
    """
    name: str
    v_eigen: np.ndarray
    v_type: Literal['absolute', 'substitution']
    labels: Tuple[str, ...]
    strength: float = field(init=False, repr=False)

    def __post_init__(self):
        """在初始化后自动计算 strength 属性并进行数据验证。"""
        # 使用 object.__setattr__ 是因为 dataclass(frozen=True) 使实例不可变
        object.__setattr__(self, 'strength', np.linalg.norm(self.v_eigen))

        # --- 数据验证增强 ---
        if self.v_type not in ['absolute', 'substitution']:
            raise ValueError(f"v_type 必须是 'absolute' 或 'substitution'，但收到了 '{self.v_type}'。")
        
        if len(self.v_eigen) != len(self.labels):
            raise ValueError(
                f"引力源 '{self.name}' 的 v_eigen 维度 ({len(self.v_eigen)}) "
                f"必须与其 labels 数量 ({len(self.labels)}) 相匹配。"
            )

        if not np.isclose(np.sum(self.v_eigen), 0):
            raise ValueError(
                f"引力源 '{self.name}' 的 v_eigen 分量之和必须接近于0 "
                f"(当前和为 {np.sum(self.v_eigen):.2e})。"
            )

    def embed(self, new_labels: List[str]) -> GravitationalSource:
        """
        通过标签匹配，将当前引力源投影到一个新的维度空间中。

        此方法采用基于“偏离模式守恒”的嵌入法，这是理论上最纯粹和
        通用的方法，能够同时处理升维、降维和维度重排。

        警告：此方法在理论上仅为 'absolute' 类型的 v_eigen 设计。
        对 'substitution' 类型的 v_eigen 使用此方法会产生不可靠的结果。

        Args:
            new_labels (List[str]): 目标宇宙的原子事件标签列表。

        Returns:
            GravitationalSource: 一个新的、被投影到新维度空间的引力源对象。
        """
        old_labels = list(self.labels)
        
        if new_labels == old_labels:
            return self

        if self.v_type == 'substitution':
            warnings.warn(
                f"警告：你正在对 'substitution' 类型的源 '{self.name}' 调用 embed()。"
                "此操作理论上不严谨，结果可能非常不可靠。",
                UserWarning
            )

        # 步骤 1: 解码旧模式并与标签关联
        p_old_dict = dict(zip(old_labels, exp_map_from_origin(self.v_eigen)))

        # 步骤 2: 在新画布上重建概率分布
        p_new_draft = np.zeros(len(new_labels))
        for i, label in enumerate(new_labels):
            if label in p_old_dict:
                p_new_draft[i] = p_old_dict[label]
        
        # 步骤 3: 重整化并编码新效应
        sum_draft = np.sum(p_new_draft)
        if sum_draft < 1e-12:
            warnings.warn(
                f"在 embed 操作中，源 '{self.name}' 的旧标签 {old_labels} 与新标签 "
                f"{new_labels} 之间没有任何重叠。将返回一个零效应源。"
            )
            v_embedded = np.zeros(len(new_labels))
        else:
            p_new_normalized = p_new_draft / sum_draft
            v_embedded = log_map_from_origin(p_new_normalized)
        
        return GravitationalSource(
            name=self.name, 
            v_eigen=v_embedded, 
            v_type=self.v_type, 
            labels=tuple(new_labels)
        )

    def to_substitution(self, reference_source: GravitationalSource) -> GravitationalSource:
        """
        将一个 'absolute' 源转换为相对于另一个 'absolute' 源的 'substitution' 源。
        """
        if self.v_type != 'absolute' or reference_source.v_type != 'absolute':
            raise TypeError("to_substitution 方法只能用于两个 'absolute' 类型的源相减。")
        if self.labels != reference_source.labels:
            raise ValueError("源和参照物的标签坐标系必须完全相同。")
            
        v_sub = self.v_eigen - reference_source.v_eigen
        new_name = f"{self.name}_vs_{reference_source.name}"
        
        return GravitationalSource(
            name=new_name, 
            v_eigen=v_sub, 
            v_type='substitution', 
            labels=self.labels
        )

    def __repr__(self) -> str:
        # 截断标签列表以避免过长的输出
        labels_repr = self.labels[:3]
        if len(self.labels) > 3:
            labels_repr = list(labels_repr) + ['...']

        return (f"GravitationalSource(name='{self.name}', type='{self.v_type}', "
                f"K={len(self.v_eigen)}, strength={self.strength:.4f}, labels={labels_repr})")

    def __add__(self, other: GravitationalSource) -> GravitationalSource:
        """重载加法运算符，用于效应的线性叠加。"""
        if self.labels != other.labels:
            raise ValueError("只有在相同标签坐标系下的引力源才能相加。")
        
        new_v_type = 'substitution' if 'substitution' in (self.v_type, other.v_type) else 'absolute'
        new_v_eigen = self.v_eigen + other.v_eigen
        new_name = f"({self.name})+({other.name})"
        
        return GravitationalSource(
            name=new_name, v_eigen=new_v_eigen, v_type=new_v_type, labels=self.labels
        )

    def __sub__(self, other: GravitationalSource) -> GravitationalSource:
        """重载减法运算符，主要用于计算替换效应。"""
        if self.labels != other.labels:
            raise ValueError("只有在相同标签坐标系下的引力源才能相减。")
        
        if self.v_type == 'absolute' and other.v_type == 'absolute':
            return self.to_substitution(other)
        
        if self.v_type == other.v_type:
            new_v_eigen = self.v_eigen - other.v_eigen
            new_name = f"({self.name})-({other.name})"
            return GravitationalSource(
                name=new_name, v_eigen=new_v_eigen, v_type=self.v_type, labels=self.labels
            )
        else:
            raise TypeError(
                f"不同类型的 v_eigen ('{self.v_type}' 和 '{other.v_type}') 不能直接相减。"
                "如果意图是计算替换效应，请确保两者都是 'absolute' 类型。"
            )

    def __mul__(self, scalar: float) -> GravitationalSource:
        """重载乘法运算符，用于缩放效应强度。"""
        if not isinstance(scalar, (int, float)):
            raise TypeError("引力源只能与标量相乘。")
            
        new_v_eigen = self.v_eigen * scalar
        new_name = f"{self.name}*{scalar:.2f}"
        
        return GravitationalSource(
            name=new_name, v_eigen=new_v_eigen, v_type=self.v_type, labels=self.labels
        )