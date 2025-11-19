# cgd/core/universe.py

from __future__ import annotations
from dataclasses import dataclass, field
# --- 核心修复：导入 Any 用于类型注解 ---
from typing import List, Tuple, TYPE_CHECKING, Optional, Literal, Any 
import numpy as np
import warnings

# --- 核心依赖 ---
from ..geometry.simplex import Simplex

if TYPE_CHECKING:
    from .source import GravitationalSource

@dataclass
class Universe:
    """
    一个封装了特定CGD宇宙物理属性和动态后端的类。
    """
    K: int
    alpha: float
    labels: Optional[Tuple[str, ...]] = None
    backend: Literal['numpy', 'jax'] = 'numpy'
    
    simplex: Simplex = field(init=False)
    
    # --- 核心修复：为内部字段添加类型注解 ---
    _finder_class: Any = field(init=False, repr=False)
    _simulator_class: Any = field(init=False, repr=False)

    def __post_init__(self):
        """
        在初始化后处理所有设置，包括动态绑定计算后端。
        """
        if self.alpha < 0:
            raise ValueError(f"alpha 值必须为非负数，但收到了 {self.alpha}。")
        self.simplex = Simplex(self.K)
        
        if self.labels is None:
            self.labels = tuple(str(i) for i in range(self.K))
        elif len(self.labels) != self.K:
            raise ValueError(f"提供的 labels 数量 ({len(self.labels)}) 必须与宇宙维度 K ({self.K}) 匹配。")
        else:
            self.labels = tuple(self.labels)

        # --- 动态绑定计算后端 ---
        # 移除了打印语句，让库的导入更干净
        # print(f"正在初始化一个使用 '{self.backend}' 后端的宇宙...")
        if self.backend == 'jax':
            try:
                # 在函数内部进行导入，避免循环依赖和不必要的加载
                from ..dynamics_jax.solvers_jax import EquilibriumFinderJax
                self._finder_class = EquilibriumFinderJax
                
                # JAX后端目前不支持轨迹模拟，所以绑定NumPy的
                from ..dynamics.solvers import TrajectorySimulator
                self._simulator_class = TrajectorySimulator
                # warnings.warn("轨迹模拟当前仅支持 'numpy' 后端。")

            except ImportError:
                raise ImportError(
                    "无法加载 JAX 后端。请确保已安装 JAX: `pip install 'jax[cpu]'` "
                    "或 `pip install 'jax[cuda]'` (for GPU)。"
                )
        elif self.backend == 'numpy':
            from ..dynamics.solvers import EquilibriumFinder, TrajectorySimulator
            self._finder_class = EquilibriumFinder
            self._simulator_class = TrajectorySimulator
        else:
            raise ValueError(f"不支持的后端: '{self.backend}'。请选择 'numpy' 或 'jax'。")

    def _validate_sources(self, sources: List['GravitationalSource']):
        """验证所有引力源的维度是否与宇宙匹配。"""
        for source in sources:
            if len(source.v_eigen) != self.K:
                raise ValueError(
                    f"引力源 '{source.name}' (K={len(source.v_eigen)}) "
                    f"与宇宙维度 (K={self.K}) 不匹配。\n"
                    f"请确保在调用 Universe 方法前，所有源都已通过 .embed() 方法"
                    f"正确投影到目标宇宙 '{self.labels}'。"
                )

    def find_equilibria(self, 
                        sources: List['GravitationalSource'], 
                        **kwargs) -> List[np.ndarray]:
        """
        在当前宇宙中，使用已配置的后端寻找平衡点。
        """
        self._validate_sources(sources)
        finder = self._finder_class(universe=self, sources=sources)
        return finder.find(**kwargs)

    def simulate_trajectory(self, 
                            p_initial: np.ndarray,
                            sources: List['GravitationalSource'], 
                            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        在当前宇宙中，模拟一个概率粒子从初始位置出发的运动轨迹。
        """
        self._validate_sources(sources)
        # 将 p_initial 作为关键字参数传递给构造函数
        simulator = self._simulator_class(universe=self, sources=sources, p_initial=p_initial, **kwargs)
        return simulator.simulate()

    def __repr__(self) -> str:
        labels_repr = self.labels[:2] if self.labels else []
        if self.labels and len(self.labels) > 2:
            labels_repr = list(labels_repr) + ['...']
        return (f"Universe(K={self.K}, alpha={self.alpha:.4f}, "
                f"backend='{self.backend}', labels={labels_repr})")