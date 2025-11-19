# CGD 测试策略与实施蓝图 (v5.0)

**报告目的**: 本文档是一份高层次的、结构化的“测试策略与实施蓝图”。它综合了之前所有阶段的分析成果，旨在为 `cgd` 代码库构建一个健壮、可维护且全面的自动化测试框架。本文档将作为后续测试代码实现的直接“任务书”。

---

## 1. 测试技术栈与文件结构 (Testing Stack & File Structure)

### 1.1. 技术栈选择

为确保测试框架的现代化、可扩展性和强大的功能，我们选择以下技术栈：

-   **测试框架**: **`pytest`**
    -   **理由**: 作为现代Python项目的事实标准，`pytest` 提供了简洁的断言语法、强大的 fixture 模型以及丰富的插件生态，是构建清晰、可读测试代码的最佳选择。
-   **辅助插件**:
    -   **`pytest-mock`**: 用于在测试中轻松地 mock 对象和函数，实现对被测单元的隔离。
    -   **`pytest-cov`**: 用于生成详细的测试覆盖率报告，确保我们的测试覆盖了绝大部分关键代码路径。

### 1.2. 文件结构规划

测试代码将存放在项目根目录下的 `tests/` 文件夹中。该文件夹将**镜像 (mirror)** `cgd/` 源文件夹的目录结构，以实现清晰的组织和直观的导航。

```
cgd-project/
├── cgd/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── source.py
│   │   └── universe.py
│   ├── dynamics/
│   │   ├── __init__.py
│   │   └── solvers.py
│   └── ...
└── tests/
    ├── __init__.py
    ├── helpers.py          # 存放核心测试辅助工具
    ├── core/
    │   ├── __init__.py
    │   ├── test_source.py    # source.py 的测试
    │   └── test_universe.py  # universe.py 的测试
    ├── dynamics/
    │   ├── __init__.py
    │   └── test_solvers.py   # solvers.py 的测试
    └── ...
```

---

## 2. 核心测试辅助工具 (Core Test Helpers)

为了提高代码复用率并简化测试用例的编写，我们将在 `tests/helpers.py` 文件中实现以下可复用的测试辅助函数。

| 函数签名 | 功能描述 |
| :--- | :--- |
| `create_test_source(p: np.ndarray, name: str, v_type: Literal['absolute', 'substitution'], labels: List[str]) -> GravitationalSource` | **安全地创建测试源**: 接收一个概率分布向量 `p`，并使用 `log_map_from_origin` 将其正确转换为 `v_eigen` 效应向量。这确保了所有测试源都符合理论基础，避免了直接构造 `v_eigen` 可能带来的不一致性。 |
| `create_test_universe(K: int, alpha: float, backend: Literal['numpy', 'jax'] = 'numpy') -> Universe` | **快速创建宇宙实例**: 一个工厂函数，用于快速生成具有指定维度 `K`、`alpha` 值和后端的 `Universe` 实例。这可以极大地减少测试代码中的重复性设置代码。 |
| `generate_mock_data(groups: Dict[str, Dict[str, int]], event_labels: List[str]) -> pd.DataFrame` | **生成模拟实验数据**: 用于生成结构化的 `pandas.DataFrame`，以模拟真实的实验计数数据。`groups` 参数允许精确控制每个实验组（如 'Stage', 'Stim_A'）中每个事件的发生次数，为工作流测试提供可控的、可复现的输入。 |

---

## 3. 测试分类与优先级 (Test Categorization & Prioritization)

为了清晰地组织测试并能够按需运行特定类型的测试，我们将使用 `pytest` 的标记 (Markers) 功能。每个测试用例将根据其目的被分配一个或多个标签。

| 标签 (Marker) | 用途描述 | 示例模块 |
| :--- | :--- | :--- |
| `@pytest.mark.core` | **核心功能测试**: 验证模块的基本功能、"happy path" 流程以及常见的边界情况。这些是最高优先级的测试，必须始终保持通过。 | `test_source.py`, `test_solvers.py` |
| `@pytest.mark.validation` | **异常与验证测试**: 验证代码的输入校验、错误处理和异常抛出机制。例如，当输入无效参数时，是否能如预期般抛出 `ValueError`。 | `test_universe.py`, `test_source.py` |
| `@pytest.mark.consistency` | **一致性测试**: 验证更高层次的系统属性。主要包括两种：**理论自洽性**（如数学变换的可逆性）和**后端一致性**（确保 NumPy 和 JAX 后端对相同输入产生相同输出）。 | `test_solvers_consistency.py`, `test_geometry.py` |
| `@pytest.mark.performance` | **性能基准测试 (可选)**: 用于对代码的关键性能路径进行基准测试，以监控和防止性能退化。 | `test_solvers.py` |

---

## 4. 关键集成测试场景 (Key Integration Test Scenarios)

集成测试是验证多个模块协同工作能力的关键。我们设计了以下两个核心端到端测试场景。

### 4.1. “黄金路径”测试 (Golden Path Test)

-   **目的**: 验证从原始数据到最终科学预测的整个工作流的正确性。
-   **位置**: `tests/workflows/test_analysis_integration.py`
-   **场景描述**:
    1.  使用 `helpers.generate_mock_data` 创建一份预定义的、理论上结果已知的模拟数据集。
    2.  调用 `run_measurement_pipeline` 函数处理该数据，生成 `source_map` 和 `alpha`。
    3.  使用上一步的输出，调用 `run_universe_analysis` 函数。
    4.  **断言**: 最终计算出的 `p_predicted` 平衡点，必须与预先计算好的“黄金标准”理论值在严格的容差内 (`np.allclose`) 相符。

### 4.2. “JAX vs NumPy”对决测试 (Showdown Test)

-   **目的**: 确保 NumPy 和 JAX 两个后端在数值计算上是等效的，保证用户可以无缝切换而不影响科学结论。
-   **位置**: `tests/dynamics/test_solvers_consistency.py`
-   **场景描述**:
    1.  使用 `helpers.create_test_source` 创建一组通用的引力源。
    2.  创建两个完全相同的 `Universe` 实例（相同的 K, alpha, sources），一个使用 `backend='numpy'`，另一个使用 `backend='jax'`。
    3.  分别调用两个 universe 实例的 `find_equilibria` 方法。
    4.  **断言**:
        -   两个后端返回的平衡点列表长度必须相等。
        -   对两个列表中的平衡点按某种范数排序后，每一对对应的 `p_predicted` 向量必须在合理的浮点误差容差内 `np.allclose`。

---

## 5. 从“问题”到“测试”的映射 (Mapping Issues to Tests)

本节将 `STEP_3` 报告中发现的关键问题，直接转化为具体的、可失败的 (failing) 测试用例设计。这些测试将作为修复这些问题的“验收标准”。

### 5.1. 问题: NumPy求解器中的去重逻辑不一致

-   **根本原因**: NumPy 后端的 `EquilibriumFinder` 使用 `np.allclose`（欧几里得距离）进行去重，而理论上更严谨的 `distance_FR`（测地线距离）才应被使用。
-   **测试用例设计 (`tests/dynamics/test_solvers.py`)**:

    ```python
    import pytest
    from unittest.mock import patch
    
    @pytest.mark.core
    @pytest.mark.consistency
    def test_numpy_uniqueness_logic_fails_on_close_points():
        """
        验证当前的 NumPy 去重逻辑会错误地合并在欧几里得空间中接近、
        但在 Fisher-Rao 空间中分离的点。
        """
        # 1. 精心构造两个p向量 p1, p2
        #    - 使得 np.allclose(p1, p2) is True
        #    - 且 distance_FR(p1, p2) > uniqueness_tolerance
        p1, p2 = construct_special_points() 
    
        # 2. Mock 内部求解器，使其强制返回这两个点
        with patch('cgd.dynamics.solvers.EquilibriumFinder._solve_cd_from_single_point', 
                   return_value=[p1, p2]):
            finder = create_test_finder(...) # 创建一个测试用的 finder
            
            # 3. 调用 find()
            unique_equilibria = finder.find()
    
            # 4. 断言：正确的实现应该返回2个解，但当前实现会错误地返回1个
            assert len(unique_equilibria) == 2, \
                "去重逻辑错误地合并了两个独立的解"
    ```

### 5.2. 问题: JAX求解器中缺失Hessian稳定性验证

-   **根本原因**: JAX 后端的 `_validate_point_stability_jax` 只检查梯度是否为零，无法区分真正的局部最小值（稳定点）和鞍点（不稳定点）。
-   **测试用例设计 (`tests/dynamics_jax/test_solvers_jax.py`)**:

    ```python
    import pytest
    
    @pytest.mark.core
    @pytest.mark.consistency
    def test_jax_stability_check_incorrectly_validates_saddle_point():
        """
        验证当前的 JAX 稳定性检查会将一个已知的鞍点错误地报告为稳定。
        """
        # 1. 构造一组源，使其在某个已知点 p_saddle 处形成鞍点
        #    - 在 p_saddle 点，梯度为零
        #    - 在 p_saddle 点，势能的Hessian矩阵具有正负特征值
        saddle_sources, p_saddle = construct_saddle_point_scenario()
        
        finder_jax = create_test_jax_finder(sources=saddle_sources)
        
        # 2. 直接调用内部的稳定性验证函数
        is_stable = finder_jax._validate_point_stability_jax(p_saddle)
        
        # 3. 断言：正确的实现（包含Hessian检查）应该返回 False
        assert not is_stable, \
            "JAX 稳定性检查未能识别出鞍点，错误地将其报告为稳定"
    ```
