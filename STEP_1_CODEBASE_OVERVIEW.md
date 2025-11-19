# CGD 代码库概览

本文档旨在为新的开发者或AI代理提供一个关于 `cgd` 代码库的宏观地图，帮助快速理解其项目结构、核心模块功能以及关键API的设计。

## 1. 项目结构树

下面是 `cgd` 库的完整文件和目录结构：

```
cgd/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── source.py
│   └── universe.py
├── dynamics/
│   ├── __init__.py
│   ├── force.py
│   ├── potential.py
│   └── solvers.py
├── dynamics_jax/
│   ├── __init__.py
│   ├── force_jax.py
│   ├── potential_jax.py
│   └── solvers_jax.py
├── geometry/
│   ├── __init__.py
│   ├── _jax_impl.py
│   ├── simplex.py
│   └── transformations.py
├── measure/
│   ├── __init__.py
│   ├── calibration.py
│   ├── discovery.py
│   ├── purification.py
│   └── refinement.py
├── utils/
│   ├── __init__.py
│   └── data_helpers.py
├── visualize/
│   ├── __init__.py
│   ├── atlas.py
│   ├── fingerprints.py
│   └── landscape.py
└── workflows/
    ├── __init__.py
    └── analysis.py
```

## 2. 模块功能摘要

以下是 `cgd` 库中每个核心Python模块的功能摘要。

### cgd.core: 核心对象定义
- **`source.py`**: 定义了 `GravitationalSource` 类，它是框架中的核心数据结构，代表一个引力源，封装了其本征效应、理论类型和坐标系。
- **`universe.py`**: 定义了 `Universe` 类，用于封装CGD模拟的“物理环境”，包括维度、相互作用强度（alpha），并能动态选择计算后端（NumPy或JAX）。

### cgd.dynamics: 动力学模拟 (NumPy后端)
- **`potential.py`**: 负责计算在概率单纯形上任意点的总高斯势能。
- **`force.py`**: 通过对势能函数进行数值求导来计算任意点所受到的合力 `F_net`。
- **`solvers.py`**: 提供了动力学系统的核心求解器，包括用于寻找稳定平衡点的 `EquilibriumFinder` 和用于模拟粒子运动轨迹的 `TrajectorySimulator`。

### cgd.dynamics_jax: 动力学模拟 (JAX后端)
- **`potential_jax.py`**: 提供了势能计算的JAX实现，可通过JIT（即时编译）进行加速。
- **`force_jax.py`**: 利用JAX的自动微分功能 (`jax.grad`)，从势能函数解析地推导出合力，实现了精确和高性能的力计算。
- **`solvers_jax.py`**: 提供了平衡点求解器的JAX版本 `EquilibriumFinderJax`，其核心优化逻辑被封装在可被JIT编译的纯函数中，以实现最高性能。

### cgd.geometry: 几何工具
- **`simplex.py`**: 提供了与概率单纯形几何属性相关的核心计算（NumPy版），如计算单纯形的中心、半径以及两点间的Fisher-Rao距离。
- **`transformations.py`**: 实现了CGD理论中最关键的几何变换：在概率单纯形空间（“位置空间”）与切空间（“效应空间”）之间进行映射的对数映射 (`log_map`) 和指数映射 (`exp_map`)。
- **`_jax_impl.py`**: 提供了所有核心几何运算的JAX实现版本，是整个JAX后端能够高性能运行的几何基础。

### cgd.measure: 测量与校准
- **`purification.py`**: 提供了 `measure_source` 函数，这是连接实验数据和CGD理论实体的核心桥梁，通过“几何净化”从原始观测中分离出引力源的纯粹效应。
- **`calibration.py`**: 提供了 `AlphaFitter` 类，用于从实验数据中校准或拟合出宇宙的最佳物理常数 `alpha`。
- **`discovery.py`**: 提供了 `ProbeFitter` 类，其核心功能是从理论预测与实验观测之间的偏差中“发现”新的、未知的引力源（探针）。
- **`refinement.py`**: 提供了 `VectorInverter` 类，用于在一个已知 `alpha` 值的宇宙中，对引力源的 `v_eigen` 进行高精度精炼，以修正非线性效应。

### cgd.utils: 辅助工具
- **`data_helpers.py`**: 提供了一系列高级辅助函数，旨在简化从多种格式的原始实验数据到CGD核心对象（如 `GravitationalSource` 和 `source_map`）的转换过程。

### cgd.visualize: 可视化工具
- **`fingerprints.py`**: 提供了 `FingerprintPlotter` 类，用于将概率向量 `p` 和效应向量 `v_eigen` 可视化为直观的雷达图（“行为指纹”和“效应指纹”）。
- **`landscape.py`**: 提供了 `LandscapePlotter` 类，专门为三维（K=3）宇宙设计，用于将势能地貌和合力场投影到二维的三角图上进行可视化。
- **`atlas.py`**: 提供了 `AtlasPlotter` 类，用于可视化宇宙分析的结果，如宇宙相图、LESA偏差图和平衡点漂移轨迹图。

### cgd.workflows: 高级工作流
- **`analysis.py`**: 提供了高级端到端的工作流函数，如 `run_measurement_pipeline`（理论构建）和 `run_universe_analysis`（理论验证与探索），封装了CGD库的多个核心功能，为用户提供了便捷的主入口。

## 3. 核心对象API概览

### `cgd.core.source.GravitationalSource`

`GravitationalSource` 是一个不可变的数据类，代表CGD宇宙中的一个“物质”实体。

**公共属性 (Attributes):**

- `name` (str): 引力源的唯一名称。
- `v_eigen` (np.ndarray): 其本征效应向量，一个和为零的numpy数组，代表其内在作用模式。
- `v_type` (Literal['absolute', 'substitution']): `v_eigen`的理论类型，决定其是否具有跨维度能力。
- `labels` (Tuple[str, ...]): 定义了 `v_eigen` 各分量对应坐标轴的原子事件标签元组。
- `strength` (float): `v_eigen`的欧几里得模长，代表其绝对内在强度。

**公共方法 (Methods):**

- `embed(new_labels: List[str]) -> GravitationalSource`: 通过标签匹配，将当前引力源投影到一个新的维度空间中。
- `to_substitution(reference_source: GravitationalSource) -> GravitationalSource`: 将一个 'absolute' 源转换为相对于另一个 'absolute' 源的 'substitution' 源。

### `cgd.core.universe.Universe`

`Universe` 类封装了一个特定CGD宇宙的物理属性和动态计算后端。

**公共属性 (Attributes):**

- `K` (int): 宇宙的维度（原子事件的数量）。
- `alpha` (float): 定义了源之间相互作用强度的物理常数。
- `labels` (Optional[Tuple[str, ...]]): 定义了宇宙坐标轴的原子事件标签元组。
- `backend` (Literal['numpy', 'jax']): 宇宙使用的计算后端。
- `simplex` (Simplex): 一个包含了当前维度下单纯形几何常数（如半径和原点）的对象。

**公共方法 (Methods):**

- `find_equilibria(sources: List['GravitationalSource'], **kwargs) -> List[np.ndarray]`: 在当前宇宙中，使用已配置的后端寻找所有（稳定的）平衡点。
- `simulate_trajectory(p_initial: np.ndarray, sources: List['GravitationalSource'], **kwargs) -> Tuple[np.ndarray, np.ndarray]`: 在当前宇宙中，模拟一个概率粒子从初始位置出发的运动轨迹。

## 4. 核心函数签名概览

以下是CGD库中一些顶层或核心函数的完整函数签名。

**1. 测量函数 (`cgd.measure.purification`)**
```python
def measure_source(
    name: str,
    p_total: np.ndarray,
    p_stage: np.ndarray,
    event_labels: List[str],
    measurement_type: Literal['absolute', 'substitution'] = 'substitution'
) -> GravitationalSource
```

**2. Alpha拟合器 (`cgd.measure.calibration.AlphaFitter`)**
```python
class AlphaFitter:
    def fit(
        self,
        alpha_range: Tuple[float, float] = (0.01, 5.0),
        grid_points: int = 20,
        n_jobs: int = -1,
        solver_options: Optional[Dict[str, Any]] = None
    ) -> float
```

**3. 探针拟合器 (`cgd.measure.discovery.ProbeFitter`)**
```python
class ProbeFitter:
    def fit(
        self,
        probe_name: str = "Probe"
    ) -> GravitationalSource
```

**4. 测量工作流 (`cgd.workflows.analysis`)**
```python
def run_measurement_pipeline(
    data: Union[pd.DataFrame, Dict[str, List[int]]],
    event_labels: List[str],
    measurement_type: Literal['absolute', 'substitution'] = 'substitution',
    group_col: Optional[str] = None,
    event_col: Optional[str] = None,
    stage_group_name: str = 'Stage',
    stim_group_prefix: str = 'Stim_',
    competition_groups: Optional[Dict[str, List[str]]] = None,
    alpha_fitter_options: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, GravitationalSource], Optional[float]]
```

**5. 宇宙分析工作流 (`cgd.workflows.analysis`)**
```python
def run_universe_analysis(
    source_map: Dict[str, GravitationalSource],
    alpha: float,
    analysis_labels: List[str],
    source_names_for_analysis: List[str],
    p_observed: Optional[np.ndarray] = None,
    validate_stability: bool = True,
    num_random_seeds: int = 30,
    plot: bool = True
)
```
