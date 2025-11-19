# CGD 代码库详细分析与测试计划

本文档为 `cgd` 代码库中的每一个 Python 文件提供了深入的分析报告和初步的单元测试计划，旨在为构建一个工业级、完备的自动化测试套件打下坚实的基础。

---

## 模块: `cgd.core`

### 文件名

`cgd/core/source.py`

### 核心功能详述

该模块定义了 `GravitationalSource` 类，它是整个 `cgd` 框架的基石。作为一个不可变的 `dataclass`，它封装了一个引力源的所有内在属性，将其从一个抽象的数学向量 (`v_eigen`) 提升为了一个拥有明确物理意义和上下文的“对象”。

-   **核心属性**:
    -   `v_eigen`: 本征效应向量，一个和为零的 `numpy` 数组，是引力源作用模式的数学表示。
    -   `v_type`: 理论类型 (`'absolute'` 或 `'substitution'`)，定义了效应的层级。`'absolute'` 源是相对于宇宙真空（均匀分布）定义的，具有跨维度嵌入的能力；`'substitution'` 源是相对于另一个源定义的，通常仅限在同一维度空间内进行比较。
    -   `labels`: 坐标系标签，一个字符串元组，为 `v_eigen` 的每个分量赋予了物理含义，是实现维度安全变换的关键。
    -   `strength`: 效应强度，即 `v_eigen` 的欧几里得范数，在初始化后自动计算。
-   **核心验证**: `__post_init__` 方法在对象创建后立即执行一系列验证，确保了数据的完整性和有效性，例如：`v_type` 必须合法、`v_eigen` 的维度必须与 `labels` 的数量匹配、`v_eigen` 的所有分量之和必须接近于零。
-   **核心方法**:
    -   `embed()`: 这是 `GravitationalSource` 最核心的功能之一。它允许一个 `'absolute'` 类型的源被安全地投影（升维、降维或重排）到一个新的维度空间中。其实现方式是：先通过 `exp_map` 将 `v_eigen` 解码回概率空间，然后基于 `labels` 匹配在新空间中重建概率分布，最后通过 `log_map` 将新概率分布重新编码为新的 `v_eigen`。
    -   `to_substitution()`: 一个便捷方法，用于从两个 `'absolute'` 类型的源计算它们之间的相对效应，生成一个 `'substitution'` 类型的源。
-   **操作符重载**: 通过重载 `+`、`-` 和 `*` 操作符，`GravitationalSource` 对象可以在相同的坐标系下进行直观的线性组合，极大地增强了代码的可读性和易用性。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.geometry.transformations`: 强依赖于此模块的 `exp_map_from_origin` 和 `log_map_from_origin` 函数。这是连接效应空间（`v_eigen`）和概率空间（`p`）的桥梁，是 `embed` 方法的理论基础。
-   **外部依赖**:
    -   `numpy`: 用于所有向量运算，是数据处理的基础。
    -   `warnings`: 用于在对 `'substitution'` 源调用 `embed` 方法时向用户发出警告。
    -   `dataclasses`, `typing`: Python标准库，用于构建类和类型注解。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
根据约束，所有测试用的 `v_eigen` 向量都必须通过 `log_map_from_origin` 从真实的概率分布 `p` 生成。将创建一个测试辅助函数 `_create_test_source(p, name, v_type, labels)` 来封装这个过程。
-   **测试用p向量**:
    -   **中心点**: `p_center = [1/3, 1/3, 1/3]`
    -   **近顶点**: `p_vertex = [0.999, 0.0005, 0.0005]`
    -   **近边界**: `p_edge = [0.499, 0.499, 0.002]`
    -   **非对称点**: `p_asym = [0.8, 0.1, 0.1]`

---

#### A. 正常功能测试 (Happy Path)

1.  **测试初始化**:
    -   **Case**: 使用由 `p_asym` 生成的 `v_eigen` 创建一个 `absolute` 类型的 `GravitationalSource`。
    -   **验证**: 对象成功创建，`strength` 属性被正确计算且大于0，`v_type` 和 `labels` 被正确赋值。
2.  **测试 `embed` 方法 - 升维**:
    -   **Case**: 将一个 K=3 (`labels=['A', 'B', 'C']`) 的源 `embed` 到一个 K=5 (`labels=['A', 'B', 'C', 'D', 'E']`) 的新空间。
    -   **验证**: 返回一个新的 `GravitationalSource` 对象，其 `v_eigen` 维度为5，`labels` 为新 `labels`。新 `v_eigen` 中与 'D', 'E' 对应的分量经过重新编码后不为零（由于归一化效应），但其主要“能量”分布在'A','B','C'上。
3.  **测试 `embed` 方法 - 降维**:
    -   **Case**: 将一个 K=5 的源 `embed` 到一个 K=3 的新空间（丢弃 'D', 'E'）。
    -   **验证**: 返回一个 K=3 的新源，其效应已经根据保留的维度进行了正确的重整化。
4.  **测试 `embed` 方法 - 维度重排**:
    -   **Case**: 将一个 K=3 (`labels=['A', 'B', 'C']`) 的源 `embed` 到一个 K=3 (`labels=['C', 'A', 'B']`) 的新空间。
    -   **验证**: 返回一个新源，其 `v_eigen` 的分量顺序与新 `labels` 的顺序相对应。
5.  **测试 `to_substitution` 方法**:
    -   **Case**: 创建两个 `absolute` 类型的源 `s1` 和 `s2`，然后调用 `s1.to_substitution(s2)`。
    -   **验证**: 返回一个新源，其 `v_type` 为 `'substitution'`，`name` 为 `"s1_vs_s2"`，`v_eigen` 等于 `s1.v_eigen - s2.v_eigen`。
6.  **测试操作符**:
    -   **Case**: 对两个 `labels` 相同的源 `s1`, `s2` 进行 `s1 + s2`, `s1 - s2`, `s1 * 2.0` 操作。
    -   **验证**: `v_eigen` 的计算结果正确，`v_type` 的继承逻辑正确（`+`/`-` 操作中只要有一个是`'substitution'`，结果就是`'substitution'`）。

#### B. 边界条件测试 (Boundary Conditions)

1.  **测试零效应源**:
    -   **Case**: 使用 `p_center` (`[1/K, ..., 1/K]`) 创建源。
    -   **验证**: 生成的 `v_eigen` 是一个接近于零的向量，`strength` 接近于0。
2.  **测试近顶点源**:
    -   **Case**: 使用 `p_vertex` (`[0.999, ...]`) 创建源。
    -   **验证**: 对象能被成功创建，`strength` 值很大，测试其在 `embed` 过程中的数值稳定性。
3.  **测试 `embed` - 无重叠标签**:
    -   **Case**: 将一个 (`labels=['A', 'B', 'C']`) 的源 `embed` 到 (`labels=['D', 'E', 'F']`)。
    -   **验证**: 系统产生一个 `UserWarning`，提示没有重叠，并返回一个零效应源（`v_eigen` 为零向量）。
4.  **测试 `embed` - 相同标签**:
    -   **Case**: 调用 `source.embed(source.labels)`。
    -   **验证**: 应该直接返回 `self` 对象，而不是创建一个新对象。

#### C. 异常与验证测试 (Error & Validation)

1.  **测试初始化 `ValueError` - 维度不匹配**:
    -   **Case**: 创建源时，传入长度为3的 `v_eigen` 和长度为4的 `labels`。
    -   **验证**: 抛出 `ValueError`，并包含清晰的错误信息。
2.  **测试初始化 `ValueError` - `v_eigen` 和不为零**:
    -   **Case**: （此测试需手动构造 `v_eigen`）创建一个 `v_eigen`，其分量之和不为零。
    -   **验证**: 抛出 `ValueError`。
3.  **测试初始化 `ValueError` - 非法 `v_type`**:
    -   **Case**: 创建源时，传入 `v_type='invalid_type'`。
    -   **验证**: 抛出 `ValueError`。
4.  **测试 `embed` - `UserWarning`**:
    -   **Case**: 对一个 `'substitution'` 类型的源调用 `embed` 方法。
    -   **验证**: 捕获并验证 `UserWarning` 被正确触发。
5.  **测试操作符 `ValueError` - 坐标系不匹配**:
    -   **Case**: 对两个 `labels` 不同的源进行 `+` 或 `-` 操作。
    -   **验证**: 抛出 `ValueError`。
6.  **测试 `to_substitution` 的类型和值错误**:
    -   **Case 1**: 对一个 `'absolute'` 和一个 `'substitution'` 源调用此方法。
    -   **验证 1**: 抛出 `TypeError`。
    -   **Case 2**: 对两个 `labels` 不同的 `'absolute'` 源调用此方法。
    -   **验证 2**: 抛出 `ValueError`。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **测试 `embed` 方法的可逆性**:
    -   **Case**:
        1.  创建一个 K=3, `labels=['A', 'B', 'C']` 的源 `s_original`。
        2.  将其升维至 K=5, `labels=['A', 'B', 'C', 'D', 'E']`，得到 `s_embedded`。
        3.  再将其降维回 K=3, `labels=['A', 'B', 'C']`，得到 `s_reverted`。
    -   **验证**: `s_reverted.v_eigen` 与 `s_original.v_eigen` 在数值上应该高度接近 (`np.allclose`)。
2.  **测试线性叠加假设**:
    -   **Case**:
        1.  创建三个 `absolute` 类型的源 `sA`, `sB`, `sC`。
        2.  通过操作符计算 `s_sum = sA + sB`。
        3.  通过 `to_substitution` 计算 `s_sub_manual = sA - sC`。
        4.  通过操作符计算 `s_sub_op = sA - sC`。
    -   **验证**: `s_sum.v_eigen` 等于 `sA.v_eigen + sB.v_eigen`。 `s_sub_manual.v_eigen` 等于 `s_sub_op.v_eigen`，并且它们的 `v_type` 都是 `'substitution'`。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: `GravitationalSource` 类本身是纯 NumPy 实现。然而，它的核心方法 `embed` 依赖于 `cgd.geometry` 模块。因此，对此模块的后端一致性测试将间接覆盖 `embed` 方法所依赖的计算。在为 `cgd/geometry/transformations.py` 制定测试计划时，将包含对 `exp_map` 和 `log_map` 的严格跨后端一致性验证。

---
### 文件名

`cgd/core/universe.py`

### 核心功能详述

该模块定义了 `Universe` 类，它是所有 `cgd` 动力学模拟的中心“环境”或“舞台”。它不仅封装了宇宙的物理定律（维度`K`和相互作用强度`alpha`），更重要的是，它扮演了计算后端的**动态调度器 (Dynamic Dispatcher)** 角色。

-   **核心功能**:
    -   **物理参数化**: `Universe` 对象持有定义一个模拟世界所需的所有核心参数：维度`K`、相互作用强度`alpha`、以及坐标系`labels`。
    -   **后端动态绑定**: 这是该类的关键架构特性。在 `__post_init__` 初始化阶段，它会根据用户传入的 `backend` 字符串（`'numpy'` 或 `'jax'`）动态地导入并绑定相应的求解器类（`EquilibriumFinder` 或 `EquilibriumFinderJax`）。这使得上层代码可以通过一个简单的字符串标志无缝切换整个计算引擎，而无需关心底层的实现细节。
    -   **高级API**: 它提供了两个核心的公共方法，`find_equilibria()` 和 `simulate_trajectory()`。这两个方法是用户与动力学系统交互的主要入口。它们内部会将计算任务委托给已经绑定的后端求解器实例来完成。
    -   **维度安全验证**: 在执行任何计算之前，`_validate_sources` 方法会进行一次关键的预检查，确保所有传入的 `GravitationalSource` 对象的维度都与宇宙自身的维度`K`相匹配。这个机制可以有效防止因源未被正确`embed`而导致的运行时错误。
    -   **自身参数验证**: 在初始化时，它还会对自身的参数进行验证，例如`alpha`必须为非负数，`labels`的长度必须与`K`匹配等，保证了宇宙实例的有效性。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.geometry.simplex.Simplex`: 用于创建一个 `simplex` 属性，便捷地访问该维度下的几何常数（如原点和半径）。
    -   `cgd.core.source.GravitationalSource`: 用于类型注解和在 `_validate_sources` 方法中进行实例检查。
    -   `cgd.dynamics.solvers`: 动态导入 `EquilibriumFinder` 和 `TrajectorySimulator`，作为NumPy后端的实现。
    -   `cgd.dynamics_jax.solvers_jax`: 动态导入 `EquilibriumFinderJax`，作为JAX后端的实现。
-   **外部依赖**:
    -   `numpy`: 用于基础数组操作。
    -   `typing`, `dataclasses`: Python标准库，用于构建类和类型注解。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   将使用上一节中定义的 `_create_test_source` 辅助函数创建测试用的`GravitationalSource`对象。
-   `alpha` 的测试值将严格遵守 `[0.0, 2.0]` 的约束。

---

#### A. 正常功能测试 (Happy Path)

1.  **测试初始化与后端绑定**:
    -   **Case**: 创建一个 `backend='numpy'` 的 `Universe`。
    -   **验证**: 实例的 `_finder_class` 属性应指向 `cgd.dynamics.solvers.EquilibriumFinder`。
    -   **Case**: 创建一个 `backend='jax'` 的 `Universe`。
    -   **验证**: 实例的 `_finder_class` 属性应指向 `cgd.dynamics_jax.solvers_jax.EquilibriumFinderJax`。
2.  **测试`labels`的自动生成**:
    -   **Case**: 初始化时不提供 `labels` 参数，`K=3`。
    -   **验证**: 实例的 `labels` 属性应为 `('0', '1', '2')`。
3.  **测试API调度逻辑 (使用Mocks)**:
    -   **Case**: 创建一个NumPy后端宇宙，并调用 `find_equilibria`。
    -   **验证**: 使用 `unittest.mock` 确认 `EquilibriumFinder` 的 `find` 方法被正确调用。这可以隔离地测试调度逻辑，而不必实际运行耗时的计算。
    -   **Case**: 调用 `simulate_trajectory`。
    -   **验证**: 使用 `unittest.mock` 确认 `TrajectorySimulator` 的 `simulate` 方法被正确调用。

#### B. 边界条件测试 (Boundary Conditions)

1.  **测试最小维度**:
    -   **Case**: 使用 `K=1` 初始化 `Universe`。
    -   **验证**: 宇宙应能成功创建，其 `simplex.radius` 应为 `0.0`。
2.  **测试 `alpha` 参数范围**:
    -   **Case**: 分别使用 `alpha=0.0` 和 `alpha=2.0` 初始化宇宙。
    -   **验证**: 两种情况都应能成功创建。
3.  **测试无引力源的场景**:
    -   **Case**: 创建一个宇宙（`alpha=1.0`），然后调用 `find_equilibria` 并传入一个空的 `sources` 列表 (`[]`)。
    -   **验证**: 函数应能正常执行并返回一个包含单个元素的列表，该元素为宇宙的中心点（混沌原点）。

#### C. 异常与验证测试 (Error & Validation)

1.  **测试初始化 `ValueError`**:
    -   **Case**: 使用负数 `alpha` (例如 `-1.0`) 初始化。
    -   **验证**: 抛出 `ValueError`。
    -   **Case**: 传入的 `labels` 数量与 `K` 不匹配。
    -   **验证**: 抛出 `ValueError`。
    -   **Case**: 传入一个无效的 `backend` 字符串 (例如 `'torch'`)。
    -   **验证**: 抛出 `ValueError`。
2.  **测试 `ImportError` (当JAX未安装时)**:
    -   **Case**: （需要使用 `mock` 来模拟 `import jax` 失败）在JAX未安装的环境下，使用 `backend='jax'` 初始化。
    -   **验证**: 应该捕获到一个 `ImportError`，且错误信息应明确提示用户安装JAX。
3.  **测试源维度验证**:
    -   **Case**: 创建一个 `K=3` 的宇宙，但向 `find_equilibria` 传入一个 `K=4` 的引力源。
    -   **验证**: 应该从 `_validate_sources` 中抛出一个内容清晰的 `ValueError`。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **测试 `alpha=0` 时的LESA一致性**:
    -   **Case**:
        1.  创建一个 `K=3` 的 `absolute` 源 `s1`。
        2.  创建一个 `alpha=0.0` 的宇宙 `u`。
        3.  调用 `u.find_equilibria(sources=[s1])` 得到平衡点 `p_eq`。
        4.  直接计算LESA的预测 `p_lesa = exp_map_from_origin(s1.v_eigen)`。
    -   **验证**: `p_eq[0]` 和 `p_lesa` 应该在数值上高度一致 (`np.allclose`)。这验证了在没有相互作用时，动力学模拟的结果退化为纯几何映射，符合理论预期。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

1.  **测试 `find_equilibria` 结果一致性**:
    -   **Case**:
        1.  创建一套相同的引力源（例如，一个 `K=3` 的宇宙中包含两个源）。
        2.  使用 `alpha=1.5` 分别创建一个NumPy后端宇宙 `u_np` 和一个JAX后端宇宙 `u_jax`。
        3.  分别调用 `u_np.find_equilibria()` 和 `u_jax.find_equilibria()` 获取两组平衡点。
    -   **验证**: 两组返回的平衡点列表，在排序后，其对应的平衡点向量在数值上应该高度一致 (`np.allclose` with a reasonable tolerance)。这是一个关键的集成测试，确保了两个后端对于相同的物理问题给出相同的答案。

---
## 模块: `cgd.geometry`

### 文件名

`cgd/geometry/simplex.py`

### 核心功能详述

该模块提供了计算概率单纯形基本几何属性的核心函数（NumPy后端）。它定义了单纯形空间的度量衡，是所有上层几何变换和动力学模拟的基础。

-   **核心功能**:
    -   `get_chaos_origin(K)`: 计算并返回给定维度`K`的单纯形的中心点，即均匀概率分布 `[1/K, 1/K, ...]`。这个点在CGD理论中是所有效应的参考原点。
    -   `get_radius(K)`: 计算给定维度`K`的单纯形的最大半径，即从中心点到任意一个顶点的Fisher-Rao测地线距离。这个值常用于距离的归一化。
    -   `distance_FR(p, q)`: 计算两个概率分布`p`和`q`之间的Fisher-Rao距离。这是单纯形流形上的“直线距离”，是定义势能和比较点与点之间差异的基础度量。函数内部包含了对输入的归一化和数值稳定性处理（如`clip`）。
    -   **`Simplex`类**: 这是一个便捷的`dataclass`，封装了特定维度`K`的几何常数。它通过属性（`.origin`, `.radius`）提供了对上述函数的缓存调用，避免了重复计算，并简化了`Universe`类中的代码。
-   **性能优化**: `get_chaos_origin` 和 `get_radius` 两个函数都使用了 `functools.lru_cache` 装饰器，可以缓存常用`K`值的结果，在多次调用时显著提高性能。

### 依赖关系分析

-   **内部依赖**: 无。这是一个基础模块。
-   **外部依赖**:
    -   `numpy`: 用于所有向量运算。
    -   `functools`, `dataclasses`: Python标准库。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   将使用 `numpy` 创建一系列标准概率向量作为测试输入。
    -   `p_3d_center = [1/3, 1/3, 1/3]`
    -   `p_3d_v0 = [1.0, 0.0, 0.0]`
    -   `p_3d_v1 = [0.0, 1.0, 0.0]`
    -   `p_3d_asym = [0.8, 0.1, 0.1]`
    -   `p_unnormalized = [1, 2, 3]` (未归一化向量)

---

#### A. 正常功能测试 (Happy Path)

1.  **`get_chaos_origin`**:
    -   **Case**: 调用 `get_chaos_origin(K=4)`。
    -   **验证**: 返回 `[0.25, 0.25, 0.25, 0.25]`。
2.  **`get_radius`**:
    -   **Case**: 调用 `get_radius(K=3)`。
    -   **验证**: 返回 `2 * np.arccos(np.sqrt(1/3))`，约等于 `1.91`。
3.  **`distance_FR`**:
    -   **Case**: 计算 `distance_FR(p_3d_center, p_3d_v0)`。
    -   **验证**: 结果应等于 `get_radius(K=3)`。
    -   **Case**: 计算 `distance_FR(p_3d_v0, p_3d_v1)`。
    -   **验证**: 结果应为 `2 * np.arccos(0) = pi`，即从一个顶点到另一个顶点的距离。
4.  **`Simplex` 类**:
    -   **Case**: 创建 `s = Simplex(K=3)`。
    -   **验证**: `np.allclose(s.origin, get_chaos_origin(3))` 并且 `s.radius == get_radius(3)`。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`get_chaos_origin` / `get_radius` - 最小维度**:
    -   **Case**: 调用 `get_chaos_origin(K=1)` 和 `get_radius(K=1)`。
    -   **验证**: `get_chaos_origin(1)` 返回 `[1.0]`。`get_radius(1)` 返回 `0.0`。
2.  **`distance_FR` - 相同点**:
    -   **Case**: 计算 `distance_FR(p_3d_asym, p_3d_asym)`。
    -   **验证**: 距离应严格为 `0.0`。
3.  **`distance_FR` - 未归一化输入**:
    -   **Case**: 计算 `distance_FR(p_unnormalized, p_3d_center)`。
    -   **验证**: 函数应能自动处理归一化并返回一个有效距离，而不是抛出错误。
4.  **`distance_FR` - 包含微小负值的输入 (浮点误差)**:
    -   **Case**: `p = [1.0, -1e-12, 0.0]`。
    -   **验证**: 函数应能通过内部的 `p[p < 0] = 0` 逻辑处理这种情况，并正常返回结果，而不是因 `sqrt` 域名错误而失败。

#### C. 异常与验证测试 (Error & Validation)

1.  **`get_chaos_origin` / `get_radius` - 无效维度**:
    -   **Case**: 调用 `get_chaos_origin(K=0)` 或 `get_radius(K=-1)`。
    -   **验证**: 两种情况都应抛出 `ValueError`。
2.  **`distance_FR` - 维度不匹配**:
    -   **Case**: `p` 是3维向量，`q` 是4维向量。
    -   **验证**: 抛出 `ValueError`。

#### D. 理论自洽性测试 (Theoretical Consistency)

-   **说明**: 本模块的函数是理论的定义本身，因此自洽性测试主要体现在与其他模块的交互中。例如，验证 `log_map_from_origin(p)` 的范数是否等于 `distance_FR(get_chaos_origin(K), p)`。此类测试将在 `transformations.py` 的计划中详述。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 这是本模块测试计划的**核心**部分。
1.  **`get_chaos_origin` 一致性**:
    -   **Case**: 对一系列 `K` 值（例如 `K=2, 3, 10`），分别调用 `get_chaos_origin(K)` 和 `get_chaos_origin_jax(K)`。
    -   **验证**: 确保 `np.allclose()` 对两个函数的返回结果成立。
2.  **`distance_FR` 一致性**:
    -   **Case**: 使用多种输入组合（`p_center` vs `p_vertex`, `p_vertex` vs `p_asym` 等）分别调用 `distance_FR` 和 `distance_FR_jax`。
    -   **验证**: 确保两个函数返回的标量距离在很高的精度下是相等的。
    -   **Case (边界)**: 使用包含零值的顶点向量进行测试，以确保JAX版本中的 `jnp.maximum(p, 0)` 逻辑与NumPy版本等效。

---
### 文件名

`cgd/geometry/transformations.py`

### 核心功能详述

该模块是 `cgd` 理论的几何核心，提供了在两个关键数学空间之间进行转换的函数：**概率单纯形空间**（弯曲的、非线性的“位置空间”）和其在原点的**切空间**（平坦的、线性的“效应空间”）。这种转换能力使得在线性空间中进行直观的向量运算（如叠加、相减）成为可能。

-   **核心功能**:
    -   `log_map_from_origin(p)`: **对数映射**。此函数接收一个单纯形上的概率点 `p`，并计算出从原点 `O` 指向该点的切向量 `v`。这个向量 `v` 就是 `GravitationalSource` 的 `v_eigen`，它完全捕捉了从原点 `O` 到 `p` 的方向和距离信息，并且存在于一个欧几里得空间中。
    -   `exp_map_from_origin(v)`: **指数映射**。此函数接收一个切空间中的效应向量 `v`，并计算出从原点 `O` 沿着该向量方向在单纯形流形上行进相应距离后到达的终点 `p`。这是将抽象的“效应”转化为具体的“概率分布”的过程。
-   **数值稳定性**: 两个函数都包含了对边界情况的检查（例如，输入点 `p` 与原点重合，或输入向量 `v` 是零向量），并通过 `EPSILON` 常量来处理浮点数精度问题，避免除以零等错误。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.geometry.simplex.get_chaos_origin`: 强依赖于此函数来获取所有计算的参考原点 `O`。
-   **外部依赖**:
    -   `numpy`: 用于所有向量运算。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   将继续使用上一节中定义的标准概率向量 (`p_3d_center`, `p_3d_vertex`, `p_3d_asym` 等) 作为 `log_map_from_origin` 的输入。
-   对于 `exp_map_from_origin`，其输入 `v` 向量将通过调用 `log_map_from_origin` 从上述 `p` 向量生成，以确保测试数据的真实性和相关性。

---

#### A. 正常功能测试 (Happy Path)

1.  **`log_map_from_origin`**:
    -   **Case**: 使用非对称点 `p_3d_asym` 调用函数。
    -   **验证**: 返回一个 `ndarray`，其所有分量之和应非常接近于零 (`np.isclose(np.sum(v), 0)`)。
2.  **`exp_map_from_origin`**:
    -   **Case**: 从 `p_3d_asym` 生成 `v_asym`，然后调用 `exp_map_from_origin(v_asym)`。
    -   **验证**: 返回的概率向量 `p_reconstructed` 应该与原始的 `p_3d_asym` 高度一致 (`np.allclose`)。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`log_map_from_origin` - 原点输入**:
    -   **Case**: 调用 `log_map_from_origin(p_3d_center)`。
    -   **验证**: 应返回一个零向量 `[0, 0, 0]`。
2.  **`log_map_from_origin` - 顶点输入**:
    -   **Case**: 调用 `log_map_from_origin(p_3d_v0)`，其中 `p_3d_v0 = [1.0, 0.0, 0.0]`。
    -   **验证**: 应返回一个有效的、和为零的向量。
3.  **`exp_map_from_origin` - 零向量输入**:
    -   **Case**: 调用 `exp_map_from_origin(np.zeros(3))`。
    -   **验证**: 应返回单纯形的原点 `p_3d_center` (`[1/3, 1/3, 1/3]`)。
4.  **`exp_map_from_origin` - 大范数向量**:
    -   **Case**: 从一个极靠近顶点的 `p` (`[1-2*eps, eps, eps]`) 生成 `v`，其范数会很大。然后调用 `exp_map_from_origin`。
    -   **验证**: 函数应能处理大的 `norm_u` 并返回一个有效的概率分布，测试三角函数的数值稳定性。

#### C. 异常与验证测试 (Error & Validation)

-   **说明**: 这两个函数在设计上是相当健壮的，它们不预期会抛出 `ValueError` 等异常，而是通过返回零向量或原点来处理边界情况。因此，此部分的测试重点是验证这些处理逻辑是否按预期工作，这已在“边界条件测试”中覆盖。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **映射的可逆性 (核心测试)**:
    -   **Case**: 对一系列不同的 `p` 向量（中心、顶点、边界、非对称），执行 `p_reverted = exp_map_from_origin(log_map_from_origin(p))`。
    -   **验证**: 在所有情况下，`p_reverted` 都应与原始的 `p` 高度一致 (`np.allclose`)。这是验证两个核心几何变换互为逆运算的关键测试。
2.  **向量范数与几何距离的一致性**:
    -   **Case**:
        1.  选择一个点 `p_asym`。
        2.  计算其效应向量 `v = log_map_from_origin(p_asym)`。
        3.  计算 `v` 在切空间中的范数：`norm_v = np.linalg.norm(v * np.sqrt(get_chaos_origin(3)))`。（注意需要乘以`sqrt(origin)`进行坐标变换）。
        4.  计算 `p_asym` 与原点之间的Fisher-Rao距离 `dist = distance_FR(p_asym, get_chaos_origin(3))`。
    -   **验证**: `norm_v` 应等于 `dist / 2.0`。这验证了对数映射保留了距离信息，符合流形几何的理论。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 这是本模块测试计划的**核心**部分。
1.  **`log_map_from_origin` 一致性**:
    -   **Case**: 对一系列 `p` 向量（中心、顶点、非对称等）分别调用 `log_map_from_origin` (NumPy) 和 `log_map_from_origin_jax` (JAX)。
    -   **验证**: 两个函数返回的 `v` 向量在数值上应高度一致 (`np.allclose`)。
2.  **`exp_map_from_origin` 一致性**:
    -   **Case**: 从一系列 `p` 向量生成 `v` 向量，然后分别调用 `exp_map_from_origin` (NumPy) 和 `exp_map_from_origin_jax` (JAX)。
    -   **验证**: 两个函数返回的 `p` 向量在数值上应高度一致 (`np.allclose`)。
3.  **边界情况一致性**:
    -   **Case**: 使用中心点 `p_center` 和零向量 `v_zero` 作为输入，分别对两个后端的映射函数进行测试。
    -   **验证**: 确保 JAX 版本中的 `jax.lax.cond` 逻辑与 NumPy 版本中的 `if` 语句产生了完全相同的结果（零向量或原点）。

---
### 文件名

`cgd/geometry/_jax_impl.py`

### 核心功能详述

该模块提供了所有核心几何运算的 **JAX 后端实现**。它与 `simplex.py` 和 `transformations.py` 中的 NumPy 实现是一一对应的，但其设计完全遵循 JAX 的编程范式，以实现高性能计算、自动微分和即时编译（JIT）。

-   **核心功能**:
    -   `get_chaos_origin_jax(K)`: JAX 版本的 `get_chaos_origin`。
    -   `distance_FR_jax(p, q)`: JAX 版本的 `distance_FR`。
    -   `log_map_from_origin_jax(p)`: JAX 版本的 `log_map_from_origin`。
    -   `exp_map_from_origin_jax(v)`: JAX 版本的 `exp_map_from_origin`。
-   **JAX 特性应用**:
    -   **即时编译 (JIT)**: 所有函数都被 `@jit` 装饰器包裹，JAX 会在首次调用时将这些 Python 函数编译成高效的 XLA 优化机器码，极大地加速了后续的重复计算。
    -   **函数式编程**: 代码中使用了 `jax.lax.cond` 来替代标准的 `if/else` 语句。这是 JAX 的要求，因为在编译后的代码中，分支必须是“纯函数式”的，不能有 Python 运行时的副作用。这种结构使得对边界条件（如零向量）的处理能够被 JIT 编译器理解和优化。
    -   **静态参数**: 在 `get_chaos_origin_jax` 中，维度 `K` 被声明为 `@partial(jit, static_argnames=('K',))`。这向 JIT 编译器提示 `K` 是一个在编译时固定的常量，允许 JAX 生成针对特定维度 `K` 的更优化的代码。在其他函数中，`K` 是通过输入数组的 `.shape[0]` 动态推断的，这也是 JAX 的标准做法。
    -   **数值类型**: 所有操作都使用 `jax.numpy` (`jnp`)，这是 JAX 提供的与 NumPy API 兼容的库，但其数组和操作都支持在 GPU/TPU 上运行并可被自动微分。

### 依赖关系分析

-   **内部依赖**: 无。这是一个独立的 JAX 实现模块。
-   **外部依赖**:
    -   `jax` 和 `jax.numpy`: 模块的核心依赖，提供了 JIT 编译、函数式控制流和 JAX 数组对象。

### 初步测试计划 (Test Plan)

-   **说明**: 本模块的测试目标**不是**重复验证几何算法的正确性（这已在 `simplex.py` 和 `transformations.py` 的测试计划中覆盖），而是**专注于验证 JAX 实现本身的技术特性和与 NumPy 版本的行为一致性**。
-   所有具体的**一致性测试用例**已经在 `simplex.py` 和 `transformations.py` 的测试计划中的 **"E. JAX与NumPy后端一致性测试"** 部分详细定义。本计划将专注于 JAX 特有的测试。

---

#### A. 正常功能测试 (Happy Path)

1.  **测试 JIT 编译**:
    -   **Case**: 对 `distance_FR_jax` 函数进行计时测试。首次调用，然后在一个循环中多次调用。
    -   **验证**: 首次调用的时间应显著长于后续调用的平均时间。这验证了 JIT 编译确实发生且有效。
2.  **测试函数的可微分性 (Grad)**:
    -   **Case**: 使用 `jax.grad` 来获取 `distance_FR_jax` 相对于其第一个参数 `p` 的梯度。
    -   **验证**: 能够成功计算出梯度，并且梯度是一个与 `p` 形状相同的 `DeviceArray`。这验证了函数与 JAX 的自动微分系统是兼容的。可以进一步验证在 `p=q` 时梯度为零。
    -   **Case**: 对 `exp_map_from_origin_jax` 进行同样的可微分性测试。
    -   **验证**: 能够成功计算出梯度。

#### B. 边界条件测试 (Boundary Conditions)

-   **说明**: 对边界条件的**数值结果**的测试已在 NumPy 版本的测试计划中定义，并通过一致性测试覆盖到本模块。这里的重点是验证 JAX 的**控制流** (`jax.lax.cond`) 是否正确处理了这些边界情况。
1.  **`log_map_from_origin_jax` - 原点输入**:
    -   **Case**: 输入为中心点 `p_center`。
    -   **验证**: 函数应正确路由到返回零向量的分支，并且该过程是可 JIT 编译和可微分的。
2.  **`exp_map_from_origin_jax` - 零向量输入**:
    -   **Case**: 输入为零向量 `v_zero`。
    -   **验证**: 函数应正确路由到返回原点 `origin` 的分支，并且该过程是可 JIT 编译和可微分的。

#### C. 异常与验证测试 (Error & Validation)

-   **说明**: JAX 的 JIT 编译函数在遇到形状不匹配等问题时，通常会在编译时或首次运行时抛出非常明确的 `TypeError` 或 `ConcretizationTypeError`。
1.  **测试 JIT 编译时的形状错误**:
    -   **Case**: 调用 `distance_FR_jax` 时，传入一个3维向量和一个4维向量。
    -   **验证**: JAX 应抛出一个与形状不匹配相关的错误。

#### D. 理论自洽性测试 (Theoretical Consistency)

-   **说明**: 理论自洽性测试（如映射的可逆性）已经在 `transformations.py` 的测试计划中定义。在 JAX 后端，这些测试同样适用，并且是**一致性测试**的一部分。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 这是**最高优先级**的测试。所有在 `simplex.py` 和 `transformations.py` 测试计划中标记为 "E" 部分的测试用例都必须被实现。
-   **核心目标**: 确保对于同一组数学输入，`function_numpy(...)` 和 `function_jax(...)` 的输出在数值上是等价的。这保证了用户无论选择哪个后端，都能获得相同的科学计算结果。

---
## 模块: `cgd.dynamics` & `cgd.dynamics_jax`

### 文件名

`cgd/dynamics/potential.py`

### 核心功能详述

该模块提供了 `potential_gaussian` 函数，这是 `cgd` 物理模型的**核心引擎（NumPy后端）**。它的功能是计算在单纯形空间中任意一点 `p` 所感受到的、由宇宙中所有引力源产生的总高斯势能。

-   **计算逻辑**:
    1.  遍历 `Universe` 中存在的所有 `GravitationalSource`。
    2.  对于每一个源，首先使用 `exp_map_from_origin` 将其效应向量 `v_eigen` 转换到概率单纯形上的“本征位置” `p_stim`。
    3.  使用 `distance_FR` 计算当前点 `p` 与该源的本征位置 `p_stim` 之间的几何距离。
    4.  将此距离相对于宇宙半径进行归一化。
    5.  根据高斯势能公式 `U(p) = -strength * exp(-(alpha * relative_dist)^2)` 计算该源在点 `p` 产生的势能。公式中的负号表明引力源是“吸引”的。
    6.  将所有源产生的势能进行线性叠加，得到该点的总势能 `total_potential`。

这个函数的结果是后续计算合力（通过梯度）和寻找平衡点（通过最小化势能）的基础。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.source.GravitationalSource`: 需要访问源的 `v_eigen` 和 `strength` 属性。
    -   `cgd.core.universe.Universe`: 需要访问宇宙的 `K`, `alpha`, 和 `simplex.radius` 属性。
    -   `cgd.geometry.simplex.distance_FR`: 用于计算点与源之间的距离。
    -   `cgd.geometry.transformations.exp_map_from_origin`: 用于将源的 `v_eigen` 转换为其在单纯形上的位置。
-   **外部依赖**:
    -   `numpy`: 用于所有数值计算。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   创建一个 `K=3` 的 `Universe` 实例作为标准测试环境。`alpha` 的值将在 `[0.0, 2.0]` 范围内选取。
-   使用 `_create_test_source` 辅助函数创建测试用的 `GravitationalSource` 对象。
-   定义标准测试点 `p_test = [0.6, 0.2, 0.2]` 和宇宙中心点 `p_center = [1/3, 1/3, 1/3]`。

---

#### A. 正常功能测试 (Happy Path)

1.  **单源势能计算**:
    -   **Case**: 宇宙中只有一个源 `s1`，计算 `potential_gaussian(p_test, [s1], universe)`。
    -   **验证**: 结果应为一个负数。
2.  **多源势能叠加**:
    -   **Case**: 宇宙中有两个源 `s1`, `s2`。分别计算 `U1 = potential_gaussian(p_test, [s1], universe)` 和 `U2 = potential_gaussian(p_test, [s2], universe)`，然后计算 `U_total = potential_gaussian(p_test, [s1, s2], universe)`。
    -   **验证**: `U_total` 应约等于 `U1 + U2`，验证势能的线性叠加原理。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`alpha = 0` (无相互作用)**:
    -   **Case**: 设置 `universe.alpha = 0.0`。
    -   **验证**: `potential_gaussian` 的结果应等于所有源强度之和的负数 (`-sum(s.strength for s in sources)`)。因为此时 `exp(0) = 1`，势能与距离无关。
2.  **`alpha` 较大 (alpha = 2.0)**:
    -   **Case**: 设置 `universe.alpha = 2.0`。
    -   **验证**: 函数应能返回一个有效的数值结果，测试 `exp` 函数在高 `alpha` 时的数值稳定性。
3.  **无引力源**:
    -   **Case**: 调用 `potential_gaussian` 时传入一个空的 `sources` 列表。
    -   **验证**: 返回的势能应严格为 `0.0`。
4.  **点在源的本征位置**:
    -   **Case**: 创建一个源 `s1`，其 `p_stim = exp_map_from_origin(s1.v_eigen)`。计算 `potential_gaussian(p_stim, [s1], universe)`。
    -   **验证**: 此时 `dist` 为0，`exp` 项为1，结果应为 `-s1.strength`。这是势能的最小值点。
5.  **K=1 的宇宙**:
    -   **Case**: 创建一个 `K=1` 的宇宙。
    -   **验证**: 宇宙半径为0，函数应直接返回 `0.0`，避免除以零的错误。

#### C. 异常与验证测试 (Error & Validation)

-   **说明**: `potential_gaussian` 函数本身不包含显式的输入验证逻辑（这些验证由 `Universe` 和 `GravitationalSource` 的构造函数处理）。因此，测试重点是确保它在接收到有效但可能导致数值问题的输入时能正确工作。这部分已在“边界条件测试”中覆盖。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **势能与距离的关系**:
    -   **Case**:
        1.  宇宙中只有一个源 `s1`，其位置为 `p_stim`。
        2.  创建两个测试点 `p_close` 和 `p_far`，使得 `distance_FR(p_close, p_stim) < distance_FR(p_far, p_stim)`。
        3.  计算两点的势能 `U_close` 和 `U_far`。
    -   **验证**: `U_close` 应小于 `U_far`（因为势能是负数，所以 `U_close` 的绝对值更大）。这验证了势能随距离增加而减弱（趋向于0）的基本物理直觉。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 这是本模块测试计划的**核心**部分。
1.  **`potential_gaussian` 一致性**:
    -   **Case**:
        1.  创建一套相同的宇宙参数（`K`, `alpha`）和引力源 `sources`。
        2.  准备JAX版本的输入：将 `sources` 列表转换为 JAX 兼容的 `sources_p_stim_jax` 和 `sources_strength_jax` 数组。
        3.  使用相同的测试点 `p_test`，分别调用 `potential_gaussian` (NumPy) 和 `potential_gaussian_jax` (JAX)。
    -   **验证**: 两个函数返回的势能标量值应在很高的精度下相等。
    -   **覆盖范围**: 此测试应在不同的 `alpha` 值（0.1, 1.0, 2.0）、不同数量的 `sources`（0, 1, 3个）和不同的测试点 `p_test` 下重复执行，以确保两个后端在各种条件下都表现一致。

---
### 文件名

`cgd/dynamics_jax/potential_jax.py`

### 核心功能详述

该模块提供了 `potential_gaussian_jax` 函数，它是 `cgd` 物理模型核心引擎的 **JAX 后端实现**。其科学计算逻辑与 NumPy 版本的 `potential_gaussian` 完全相同，但实现方式针对 JAX 框架进行了深度优化。

-   **JAX 特性应用**:
    -   **数据结构**: 函数不直接接收 `GravitationalSource` 或 `Universe` 对象，而是接收预处理好的、JAX 可兼容的 `jnp.ndarray` 数组，如 `sources_p_stim` (所有源的本征位置) 和 `sources_strength` (所有源的强度)。这种设计遵循了 JAX 对纯函数的要求。
    -   **即时编译 (JIT)**: 函数被 `@partial(jit, static_argnames=("K",))` 装饰。`K` 被标记为静态参数，允许 JAX 针对特定维度生成优化代码。
    -   **函数式循环**: 遍历所有源的循环不是使用标准的 Python `for` 循环，而是使用 `jax.lax.fori_loop`。这是一个 JAX 可 ट्रेस (traceable) 的循环构造，能够被 JIT 编译器完全优化。
    -   **依赖关系**: 它依赖于 JAX 版本的几何函数 `distance_FR_jax`，确保了整个计算图都在 JAX 的生态系统内。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.geometry._jax_impl.distance_FR_jax`: 用于计算距离。
-   **外部依赖**:
    -   `jax` 和 `jax.numpy`: 核心依赖。

### 初步测试计划 (Test Plan)

-   **说明**: 本模块的测试重点是 JAX 实现的技术特性和与 NumPy 版本的高度一致性。
-   具体的**一致性测试用例**已在 `cgd/dynamics/potential.py` 的测试计划中的 **"E. JAX与NumPy后端一致性测试"** 部分详细定义，这里不再重复。

---

#### A. 正常功能测试 (Happy Path)

1.  **测试 JIT 编译**:
    -   **Case**: 计时测试，首次调用和循环中多次调用 `potential_gaussian_jax`。
    -   **验证**: 首次调用耗时应显著长于后续调用，验证 JIT 编译生效。
2.  **测试可微分性 (Grad)**:
    -   **Case**: 使用 `jax.grad` 获取 `potential_gaussian_jax` 相对于其第一个参数 `p` 的梯度。
    -   **验证**: 能够成功计算出梯度，返回一个有效的 `DeviceArray`。

#### B. 边界条件测试 (Boundary Conditions)

-   **说明**: 数值结果的测试通过与 NumPy 版本的一致性测试来覆盖。
1.  **K=1 的宇宙 (radius=0)**:
    -   **Case**: 传入 `radius=0`。
    -   **验证**: 函数内的 `jnp.maximum(radius, 1e-12)` 逻辑应能防止除以零，并返回一个有效结果。

#### C. 异常与验证测试 (Error & Validation)

-   **说明**: JIT 编译的函数在输入形状不匹配时会在编译或首次运行时抛出 JAX 特有的错误。
1.  **测试 JIT 编译时的形状错误**:
    -   **Case**: 传入的 `p` 向量与 `sources_p_stim` 中的向量维度不匹配。
    -   **验证**: JAX 应抛出一个与形状不匹配相关的错误。

#### D. 理论自洽性测试 (Theoretical Consistency)

-   **说明**: 与 NumPy 版本相同，通过一致性测试覆盖。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **核心目标**: 确保对于同一组物理输入（`p`, `sources`, `universe`），`potential_gaussian` (NumPy) 和 `potential_gaussian_jax` (JAX) 的输出完全一致。这是**最高优先级**的测试，具体测试用例见 `potential.py` 的测试计划。

---
### 文件名

`cgd/dynamics/force.py`

### 核心功能详述

该模块提供了 `calculate_F_net` 函数，其核心功能是计算在单纯形上任意一点 `p` 所感受到的**合力（NumPy后端）**。根据 `cgd` 的物理学定义，力是势能的负梯度 (`F = -∇U`)。

-   **计算方法**: 由于 `potential_gaussian` 是一个复杂的函数，直接求其解析梯度很困难。因此，该模块采用了一种**数值微分**的方法，即**中心差分法**，来近似计算梯度。
-   **核心步骤**:
    1.  对每一个维度 `i`，它在 `p` 点的邻近取两个点：`p_fwd` (向前一小步 `h`) 和 `p_bwd` (向后一小步 `h`)。
    2.  为了确保这两个点仍在单纯形上，它会对它们进行重整化。
    3.  它调用 `potential_gaussian` 计算这两点的势能。
    4.  通过公式 `(U(p+h) - U(p-h)) / (2h)` 计算出势能沿该维度的偏导数（即梯度的分量）。
    5.  所有分量计算完毕后，得到梯度向量 `grad`。
    6.  由于力向量必须位于单纯形的切空间中（即所有分量之和为零），它通过减去均值 `grad - np.mean(grad)` 的方式将梯度向量投影到该空间。
    7.  最后，取其负值 `-grad_projected`，得到最终的合力向量 `F_net`。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.source.GravitationalSource`: 传递给 `potential_gaussian`。
    -   `cgd.core.universe.Universe`: 传递给 `potential_gaussian`。
    -   `cgd.dynamics.potential.potential_gaussian`: **强依赖**。力的计算完全基于势能函数。
-   **外部依赖**:
    -   `numpy`: 用于所有向量运算。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   与 `potential.py` 测试类似，将使用一个标准的 `K=3` `Universe` 和预定义的 `GravitationalSource` 对象。
-   `alpha` 的值将在 `[0.0, 2.0]` 范围内选取。

---

#### A. 正常功能测试 (Happy Path)

1.  **力向量的基本属性**:
    -   **Case**: 在一个包含单个源的宇宙中，计算任意一点 `p_test` 的合力 `F_net`。
    -   **验证**: `F_net` 必须是一个和为零的向量 (`np.isclose(np.sum(F_net), 0)`)。
2.  **力的方向 (定性测试)**:
    -   **Case**: 宇宙中只有一个源 `s1`，其本征位置为 `p_stim`。计算宇宙中心点 `p_center` 的合力 `F_net`。
    -   **验证**: `F_net` 的方向应该大致指向 `p_stim`。这可以通过比较 `F_net` 与 `log_map_from_origin(p_stim)` 的方向（点积为正）来验证。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`alpha = 0` (无相互作用)**:
    -   **Case**: 设置 `universe.alpha = 0.0`。
    -   **验证**: 此时势能面是平坦的，梯度为零，因此 `calculate_F_net` 返回的 `F_net` 应为一个零向量。
2.  **无引力源**:
    -   **Case**: 调用 `calculate_F_net` 时传入一个空的 `sources` 列表。
    -   **验证**: 势能为零，梯度为零，`F_net` 应为一个零向量。
3.  **在平衡点 (势能极小点)**:
    -   **Case**: 宇宙中只有一个源 `s1`，其位置为 `p_stim`。计算 `calculate_F_net(p_stim, [s1], universe)`。
    -   **验证**: 在势能的极小值点，梯度应为零，因此 `F_net` 应为一个（接近）零的向量。这是验证求解器正确性的关键先决条件。
4.  **在对称点**:
    -   **Case**: 宇宙中有两个完全相同的源 `s1` 和 `s2`，分别位于 `v` 和 `-v`。计算宇宙中心点 `p_center` 的合力。
    -   **验证**: 由于对称性，合力应严格为零向量。

#### C. 异常与验证测试 (Error & Validation)

-   **说明**: 与 `potential_gaussian` 类似，此函数假设输入有效，不包含主动的异常抛出逻辑。其健壮性依赖于 `potential_gaussian` 在数值计算上的稳定性。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **力的叠加原理**:
    -   **Case**:
        1.  宇宙中有两个源 `s1`, `s2`。
        2.  分别计算 `F1 = calculate_F_net(p_test, [s1], universe)` 和 `F2 = calculate_F_net(p_test, [s2], universe)`。
        3.  计算总的力 `F_total = calculate_F_net(p_test, [s1, s2], universe)`。
    -   **验证**: 由于梯度算子的线性性，`F_total` 应约等于 `F1 + F2` (`np.allclose`)。这验证了我们实现的动力学模型符合力的叠加原理。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 这是**至关重要**的测试。JAX后端使用**自动微分**来计算精确的解析梯度，而NumPy后端使用**数值近似**。
1.  **`calculate_F_net` 一致性**:
    -   **Case**:
        1.  创建一套相同的宇宙参数（`K`, `alpha`）和引力源。
        2.  准备JAX版本的输入。
        3.  使用相同的测试点 `p_test`，分别调用 `calculate_F_net` (NumPy) 和 `calculate_F_net_jax` (JAX)。
    -   **验证**: NumPy版本的结果 `F_np` 和JAX版本的结果 `F_jax` 应该在数值上**非常接近**，但**不完全相等**。应使用 `np.allclose(F_np, F_jax, atol=1e-5)` 进行验证（`atol` 的值可能需要根据 `h` 的大小进行调整）。
    -   **覆盖范围**: 此测试应在多种条件下重复：不同的 `alpha` 值、不同的源组合、不同的测试点（包括靠近边界的点），以全面验证数值梯度与解析梯度的一致性。

---
### 文件名

`cgd/dynamics_jax/force_jax.py`

### 核心功能详述

该模块提供了 `calculate_F_net_jax` 函数，是合力计算的 **JAX 后端实现**。它展示了 JAX 框架最强大和最优雅的特性之一：**自动微分 (Automatic Differentiation)**。

-   **核心优势**: 与 NumPy 后端通过中心差分法进行**数值近似**梯度不同，JAX 后端通过 `jax.grad` 直接从 `potential_gaussian_jax` 函数**解析地**推导出其梯度函数。
-   **计算逻辑**:
    1.  在模块加载时，`grad_potential_fn = grad(potential_gaussian_jax, argnums=0)` 这一行代码就创建了一个新的、可调用的函数 `grad_potential_fn`。这个函数在数学上就是势能函数 `U(p)` 的梯度 `∇U(p)`。
    2.  `calculate_F_net_jax` 函数在被调用时，直接调用这个自动生成的 `grad_potential_fn` 来获得精确的梯度向量 `grad_vec`。
    3.  与 NumPy 版本一样，它将梯度向量投影到切空间（减去均值），然后取其负值，得到最终的合力 `F_net`。
-   **性能**: 整个 `calculate_F_net_jax` 函数被 `@jit` 编译，使其计算速度极快，并且完全在 JAX 的计算图内，可以无缝地与其他 JAX 操作（如更高阶的求导、在求解器中被编译等）结合。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.dynamics_jax.potential_jax.potential_gaussian_jax`: **强依赖**。梯度是直接从此函数推导出来的。
-   **外部依赖**:
    -   `jax` 和 `jax.numpy`: 核心依赖，提供了 `grad`, `jit` 等核心功能。

### 初步测试计划 (Test Plan)

-   **说明**: 本模块的测试重点是 JAX 实现的技术特性，以及与 NumPy 版本数值近似结果的**高度一致性**。
-   具体的**一致性测试用例**已在 `cgd/dynamics/force.py` 的测试计划中的 **"E. JAX与NumPy后端一致性测试"** 部分详细定义，这是最高优先级的测试。

---

#### A. 正常功能测试 (Happy Path)

1.  **测试 JIT 编译**:
    -   **Case**: 计时测试，首次调用和循环中多次调用 `calculate_F_net_jax`。
    -   **验证**: 首次调用耗时应显著长于后续调用。
2.  **力向量基本属性**:
    -   **Case**: 计算任意一点的 `F_net`。
    -   **验证**: 返回的 `F_net` 必须是一个和为零的 `DeviceArray` (`jnp.isclose(jnp.sum(F_net), 0)`)。

#### B. 边界条件测试 (Boundary Conditions)

-   **说明**: 数值结果的测试通过与 NumPy 版本的一致性测试来覆盖。
1.  **在平衡点**:
    -   **Case**: 在一个单源宇宙中，计算其本征位置 `p_stim` 处的力。
    -   **验证**: `calculate_F_net_jax` 应返回一个（接近）零的向量，验证 `grad` 在极值点为零。

#### C. 异常与验证测试 (Error & Validation)

-   **说明**: 与其他 JAX 模块类似，主要测试 JIT 编译时的类型和形状检查。
1.  **测试 JIT 编译时的形状错误**:
    -   **Case**: 传入的 `p` 向量与 `sources_p_stim` 中的向量维度不匹配。
    -   **验证**: JAX 应抛出一个与形状不匹配相关的错误。

#### D. 理论自洽性测试 (Theoretical Consistency)

-   **说明**: 与 NumPy 版本相同，通过一致性测试覆盖。核心的理论自洽性——力是势能的负梯度——在本模块中是由 `jax.grad` 保证的，因此测试重点是验证其与数值梯度的近似程度。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **核心目标**: 验证 `calculate_F_net` (数值梯度) 和 `calculate_F_net_jax` (解析梯度) 的结果在合理的容差范围内是一致的。
-   **测试方法**: 使用 `np.allclose(F_np, F_jax, atol=1e-5)` 进行比较。容差 `atol` 的选择是关键，它应该大于数值微分本身的误差，但又要足够小以确保两个后端计算的是同一个物理量。
-   **覆盖范围**: 此测试必须在多种条件下进行，以确保在单纯形的不同区域（中心、边界、顶点）和不同的物理参数（`alpha` 值）下，数值近似都足够接近精确解。具体测试用例见 `force.py` 的测试计划。

---
### 文件名

`cgd/dynamics/solvers.py`

### 核心功能详述

该模块提供了 `cgd` 动力学系统的核心求解器（NumPy后端）。它包含两个主要的类，分别用于解决动力学中的两大核心问题：终态（平衡点）和过程（轨迹）。

1.  **`EquilibriumFinder`**: 一个用于寻找系统中所有稳定平衡点（吸引子）的全局求解器。其策略是为了避免陷入局部最小值，从而找到全局最优解。
    -   **初始点生成**: `_generate_intelligent_guesses` 方法是其核心策略的起点。它会生成一系列高质量、多样化的初始猜测点，包括：
        -   **确定性点**: 宇宙中心点、各个顶点和边界中点。
        -   **启发式点 (LESA)**: 基于线性叠加假设（`alpha=0`时的解析解）给出的一个高质量预测点。
        -   **随机点**: 使用Sobol序列（一种准随机序列）在整个单纯形空间中进行均匀采样，以探索其他可能的吸引盆。
    -   **局部优化**: `_solve_cd_from_single_point` 方法是一个坐标下降（Coordinate Descent）求解器。它从一个初始点出发，通过轮流在各个维度上进行一维优化（使用`scipy.minimize_scalar`），迭代地寻找势能的局部最小值。
    -   **并行与去重**: `find` 方法是主入口。它使用 `joblib` 并行地从所有初始猜测点开始运行局部优化器。然后，它收集所有找到的候选点，并根据几何距离进行去重，得到最终的平衡点列表。
    -   **稳定性验证 (可选)**: 如果用户设置 `validate_stability=True`，它会进一步计算每个候选点处的梯度（力）和势能的Hessian矩阵。只有一个点的梯度（力）接近于零且Hessian矩阵是正定的（所有相关特征值非负），该点才被确认为稳定的吸引子。

2.  **`TrajectorySimulator`**: 一个常微分方程（ODE）求解器，用于模拟粒子在势能场中的运动轨迹。
    -   **核心**: 它将 `calculate_F_net` 函数作为动力学方程 `dp/dt = F(p)` 提供给 `scipy.integrate.solve_ivp`，这是一个强大、可靠的工业级ODE求解器。
    -   **功能**: 用户提供一个初始位置 `p_initial` 和模拟时长 `T`，`simulate` 方法会返回一系列时间点以及粒子在这些时间点上的位置，从而描绘出完整的运动轨迹。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.universe.Universe`, `cgd.core.source.GravitationalSource`: 需要宇宙和源对象来定义物理环境。
    -   `cgd.geometry`: `distance_FR` 用于收敛判断，`exp_map_from_origin` 用于LESA预测。
    -   `cgd.dynamics`: `potential_gaussian` 是坐标下降的目标函数，`calculate_F_net` 是轨迹模拟的动力学方程和稳定性验证的基础。
-   **外部依赖**:
    -   `numpy`: 核心数值计算。
    -   `scipy`: `optimize.minimize_scalar` 用于坐标下降，`stats.qmc` 用于生成Sobol序列，`linalg.eigh` 用于计算Hessian矩阵特征值，`integrate.solve_ivp` 用于轨迹模拟。
    -   `joblib`: 用于并行化 `find` 方法。
    -   `itertools`: 用于生成边界点的组合。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   将创建一个标准的 `K=3` `Universe` 和一套 `GravitationalSource` 对象。
-   `alpha` 的值将在 `[0.0, 2.0]` 范围内选取。

---

#### A. 正常功能测试 (Happy Path)

1.  **`EquilibriumFinder` - 单吸引子场景**:
    -   **Case**: `alpha=0.1`，宇宙中只有一个强源 `s1`。调用 `find()`。
    -   **验证**: 应返回一个包含单个平衡点的列表。该平衡点应非常接近 `s1` 的本征位置 `p_stim`。
2.  **`EquilibriumFinder` - 多吸引子场景 (双稳态)**:
    -   **Case**: `alpha=2.0`，宇宙中有两个分别位于不同顶点的强源。调用 `find()`。
    -   **验证**: 应返回一个包含两个平衡点的列表，它们分别靠近两个源的本征位置。
3.  **`EquilibriumFinder` - 稳定性验证**:
    -   **Case**: 在双稳态场景下，调用 `find(validate_stability=True)`。
    -   **验证**: 应只返回稳定的解。同时设计一个场景，其中存在鞍点（例如，两个源在两顶点，但在宇宙中心点梯度为零但Hessian矩阵非正定），验证该鞍点被正确过滤掉。
4.  **`TrajectorySimulator`**:
    -   **Case**: `alpha=1.0`，单源宇宙。从宇宙中心点开始模拟轨迹。
    -   **验证**: 返回的时间 `t` 和轨迹 `p(t)` 数组形状正确。轨迹的终点 `p(T)` 应接近于由 `EquilibriumFinder` 找到的平衡点。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`EquilibriumFinder` - `alpha=0`**:
    -   **Case**: `alpha=0`，宇宙中有多个源。调用 `find()`。
    -   **验证**: 应返回一个解，该解应与所有源的 `v_eigen` 线性叠加后的LESA预测完全一致。
2.  **`EquilibriumFinder` - 无引力源**:
    -   **Case**: `sources=[]`。调用 `find()`。
    -   **验证**: 应返回一个解，即宇宙的中心点。
3.  **`TrajectorySimulator` - 从平衡点开始**:
    -   **Case**: 先用 `EquilibriumFinder` 找到一个平衡点 `p_eq`。然后从 `p_eq` 开始模拟轨迹。
    -   **验证**: 轨迹应几乎保持在 `p_eq` 不动（由于数值误差可能有微小漂移）。

#### C. 异常与验证测试 (Error & Validation)

1.  **`EquilibriumFinder` - LESA失败时的回退**:
    -   **Case**: （使用 `mock` 模拟 `source.embed()` 抛出异常）在 `_generate_intelligent_guesses` 中触发异常。
    -   **验证**: 应当捕获 `UserWarning`，并且求解器应能继续使用其他初始点正常运行。
2.  **`EquilibriumFinder` - 无稳定解**:
    -   **Case**: （需要构造一个只有排斥源的特殊场景，或`alpha`极大）当 `find(validate_stability=True)` 运行时，所有候选点都不是稳定的。
    -   **验证**: 函数应返回一个空列表 `[]`，而不是抛出错误。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **`find` 结果与 `calculate_F_net` 的一致性**:
    -   **Case**: 对 `EquilibriumFinder.find()` 返回的每一个平衡点 `p_eq`。
    -   **验证**: `np.linalg.norm(calculate_F_net(p_eq, sources, universe))` 的结果都应非常接近于零。这验证了求解器找到的点确实是力为零的点。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

1.  **`EquilibriumFinder` 结果一致性**:
    -   **Case**:
        1.  创建一套完全相同的宇宙参数和引力源。
        2.  分别初始化NumPy版和JAX版的 `EquilibriumFinder`。
        3.  分别调用 `find()` 方法获取两组平衡点。
    -   **验证**: 返回的两组平衡点列表，在排序和去重后，应在数值上高度一致 (`np.allclose`)。
    -   **覆盖范围**: 此测试应在多种 `alpha` 值和源配置（单稳态、多稳态）下进行，以确保两个后端的优化算法能够可靠地收敛到相同的结果。
2.  **`TrajectorySimulator` 一致性 (未来)**:
    -   **说明**: 当前 `Universe` 类中，JAX后端会回退到使用NumPy版的 `TrajectorySimulator`。因此，目前没有JAX版的轨迹模拟器可供比较。
    -   **测试计划**: 如果未来实现了 `TrajectorySimulatorJax`，需要设计一个测试用例，比较两个后端在给定相同初值和时长时生成的轨迹 `p(t)` 是否一致。

---
### 文件名

`cgd/dynamics_jax/solvers_jax.py`

### 核心功能详述

该模块提供了 `EquilibriumFinderJax` 类，是平衡点求解器的 **JAX 高性能后端**。它采用了**混合架构**，将计算最密集的部分隔离到可被 JIT 编译的纯函数中，而将状态管理、数据预处理和并行化等“非纯”任务保留在标准的 Python 类中。

-   **`_solve_cd_single_jax` (核心纯函数)**:
    -   这是一个完全用 JAX 函数构建的、可被 JIT 编译的坐标下降求解器。
    -   **优化算法**: 与 NumPy 版本的 `_solve_cd_from_single_point` 不同，它不使用 `scipy.minimize_scalar`。相反，它实现了一个基于**梯度下降**的一维优化器 (`find_min_on_slice_jax`)。它使用 `jax.grad` 自动获取一维目标函数的梯度，并通过简单的梯度下降步骤 (`x_next = x_current - learning_rate * g`) 来寻找最小值。
    -   **JIT 优化**: 整个迭代过程（包括外层的 `max_rounds` 循环和内层的坐标下降循环）都使用 `jax.lax.fori_loop` 构建，使得整个求解器可以被 JIT **完全编译**成一个单一的、高效的计算核。所有超参数（如迭代次数、学习率）都被设为静态参数，以实现最大程度的优化。

-   **`EquilibriumFinderJax` (Python 封装类)**:
    -   **数据预处理**: `_prepare_jax_inputs` 方法负责将用户传入的 `GravitationalSource` 对象列表转换为 JAX 核心求解器所需的扁平化 `jnp.ndarray` 数组。
    -   **任务调度**: `find` 方法的逻辑与 NumPy 版本类似：生成多样化的初始点，然后使用 `joblib` **并行地**调用一个包装器 (`_solve_from_single_point_wrapper`)。
    -   **包装器**: `_solve_from_single_point_wrapper` 是 Python 和 JAX 世界之间的桥梁。它负责将 NumPy 格式的初始点转换为 JAX 数组，调用 JIT 编译的 `_solve_cd_single_jax` 函数，然后将计算结果转换回 NumPy 数组。
    -   **稳定性验证**: `_validate_point_stability_jax` 提供了一个简化的稳定性检查。它只验证找到的点的梯度（力）是否为零，而不计算 Hessian 矩阵（因为在 JAX 中计算 Hessian 矩阵虽然可行，但会增加编译复杂性）。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.Universe`, `cgd.core.source.GravitationalSource`: 用于初始化和数据预处理。
    -   `cgd.geometry`: `exp_map_from_origin` (NumPy版) 用于数据预处理和 LESA 预测，`distance_FR` 用于去重。
    -   `cgd.dynamics_jax`: `potential_gaussian_jax` 是优化的目标函数，`calculate_F_net_jax` 用于稳定性验证。
-   **外部依赖**:
    -   `jax`, `jax.numpy`: 核心依赖。
    -   `numpy`: 用于与 Python 世界的数据交换。
    -   `joblib`: 用于并行化。
    -   `scipy.stats.qmc`: 用于生成 Sobol 序列。

### 初步测试计划 (Test Plan)

-   **说明**: 本模块的测试重点是 JAX 实现的技术特性以及与 NumPy 版本求解结果的**高度一致性**。
-   具体的**一致性测试用例**已在 `cgd/dynamics/solvers.py` 的测试计划中的 **"E. JAX与NumPy后端一致性测试"** 部分详细定义，这是最高优先级的测试。

---

#### A. 正常功能测试 (Happy Path)

1.  **测试 JIT 编译**:
    -   **Case**: 计时测试，首次调用和循环中多次调用 `_solve_from_single_point_wrapper`。
    -   **验证**: 首次调用耗时应显著长于后续调用，验证核心求解器 `_solve_cd_single_jax` 的 JIT 编译生效。
2.  **求解器功能**:
    -   **Case**: 在一个单吸引子场景 (`alpha=0.1`) 下调用 `find()`。
    -   **验证**: 应能成功找到并返回唯一的平衡点。
    -   **Case**: 在一个多吸引子场景 (`alpha=2.0`) 下调用 `find()`。
    -   **验证**: 应能成功找到并返回多个平衡点。

#### B. 边界条件测试 (Boundary Conditions)

-   **说明**: 数值结果的测试通过与 NumPy 版本的一致性测试来覆盖。
1.  **`alpha=0`**:
    -   **Case**: 设置 `alpha=0` 调用 `find()`。
    -   **验证**: 返回的解应与 LESA 预测完全一致。
2.  **无引力源**:
    -   **Case**: `sources=[]` 调用 `find()`。
    -   **验证**: 返回宇宙中心点。

#### C. 异常与验证测试 (Error & Validation)

1.  **LESA 失败时的回退**:
    -   **Case**: (使用 `mock` 模拟 `source.embed()` 抛出异常)
    -   **验证**: 应当捕获 `UserWarning`，并且求解器能继续正常运行。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **`find` 结果与 `calculate_F_net_jax` 的一致性**:
    -   **Case**: 对 `EquilibriumFinderJax.find()` 返回的每一个平衡点 `p_eq`。
    -   **验证**: `np.linalg.norm(calculate_F_net_jax(...))` 的结果都应非常接近于零。这验证了 JAX 求解器找到的点确实是 JAX 力场中力为零的点。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **核心目标**: 确保 `EquilibriumFinder` (NumPy) 和 `EquilibriumFinderJax` (JAX) 对于相同的物理问题，找到**相同数量**和**相同位置**的平衡点。
-   **测试方法**:
    1.  创建一套完全相同的 `Universe` 和 `sources`。
    2.  分别调用两个后端的 `find()` 方法。
    3.  对返回的两个平衡点列表进行排序，然后逐一比较对应的点，确保它们在数值上高度一致 (`np.allclose`)。
-   **覆盖范围**: 这是**最高优先级**的测试，必须在多种条件下进行，以确保两个后端的优化算法（一个基于 `scipy.minimize_scalar`，一个基于梯度下降）能够可靠地收敛到相同的结果。具体测试用例见 `solvers.py` 的测试计划。

---
## 模块: `cgd.measure`

### 文件名

`cgd/measure/purification.py`

### 核心功能详述

该模块提供了 `measure_source` 函数，它是 `cgd` 框架中**从实验观测到理论构建的第一个关键接口**。它的核心任务是通过一种名为“几何净化” (Geometric Purification) 的过程，从原始的概率观测数据中提取出单个刺激物（stimulus）的纯粹效应。

-   **核心逻辑**:
    1.  **输入**: 接收两个概率分布：`p_stage` (仅有背景或舞台存在时的基线观测) 和 `p_total` (在背景之上加入了目标刺激物后的总观测)。
    2.  **解码**: 使用 `log_map_from_origin` 将这两个“位置” (`p`) 转换到线性的“效应空间” (`v`)，得到 `v_stage` 和 `v_total`。
    3.  **净化**: 在效应空间中，根据线性叠加假设 (`v_total = v_stage + v_stimulus`)，通过简单的向量减法 `v_eigen = v_total - v_stage` 来分离出刺激物的纯粹效应向量 `v_eigen`。
    4.  **封装**: 将计算出的 `v_eigen` 与用户提供的元数据（`name`, `measurement_type`, `event_labels`）一起打包，构造成一个信息完备的 `GravitationalSource` 对象并返回。
-   **输入验证**: 函数在执行计算前会验证 `p_total`, `p_stage` 和 `event_labels` 的维度是否一致，确保了测量的有效性。
-   **向后兼容**: 模块中保留了一个旧的 `geometric_purification` 函数，但标记为已废弃。新接口 `measure_source` 强制用户提供元数据，大大增强了代码的安全性和可维护性。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.geometry.transformations.log_map_from_origin`: **强依赖**，是整个净化过程的几何基础。
    -   `cgd.core.source.GravitationalSource`: 用于封装和返回最终结果。
-   **外部依赖**:
    -   `numpy`: 用于向量减法。
    -   `typing`: Python标准库。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   我们将**反向**模拟实验过程来生成测试数据。
    1.  创建两个已知的 `absolute` 源 `s_stage` 和 `s_stim`。
    2.  根据线性叠加假设，`v_total = s_stage.v_eigen + s_stim.v_eigen`。
    3.  通过指数映射生成模拟的观测数据：
        -   `p_stage_sim = exp_map_from_origin(s_stage.v_eigen)`
        -   `p_total_sim = exp_map_from_origin(v_total)`
    4.  将 `p_stage_sim` 和 `p_total_sim` 作为 `measure_source` 的输入。

---

#### A. 正常功能测试 (Happy Path)

1.  **标准测量**:
    -   **Case**: 使用上述策略生成 `p_stage_sim` 和 `p_total_sim`，然后调用 `measure_source` 来测量刺激物。
    -   **验证**: 返回的 `GravitationalSource` 对象的 `v_eigen` 应该与原始的 `s_stim.v_eigen` 在数值上高度一致 (`np.allclose`)。

#### B. 边界条件测试 (Boundary Conditions)

1.  **无刺激物效应**:
    -   **Case**: `p_total` 与 `p_stage` 相同。
    -   **验证**: 测量出的 `v_eigen` 应该是一个零向量。
2.  **舞台为宇宙中心 (无背景)**:
    -   **Case**: `p_stage` 是宇宙的中心点 `[1/K, ..., 1/K]`。
    -   **验证**: 测量出的 `v_eigen` 应该等于 `log_map_from_origin(p_total)`。

#### C. 异常与验证测试 (Error & Validation)

1.  **维度不匹配**:
    -   **Case**: 传入的 `p_total` (3维), `p_stage` (3维) 与 `event_labels` (4个) 维度不匹配。
    -   **验证**: 抛出 `ValueError`。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **测量的可逆性**:
    -   **Case**:
        1.  创建源 `s_stage` 和 `s_stim`。
        2.  生成模拟观测数据 `p_stage` 和 `p_total`。
        3.  调用 `s_measured = measure_source(...)`。
    -   **验证**: `np.allclose(s_measured.v_eigen, s_stim.v_eigen)`。这本质上是 A.1 的重申，但它从理论一致性的角度强调了“测量”是“效应叠加”的逆过程。
2.  **替换效应的线性可加性 (Substitution Additivity)**:
    -   **Case**: (这是一个关键的理论验证)
        1.  创建三个 `absolute` 源 `sA`, `sB`, `sC` (代表刺激物 A, B 和对照组 C)。
        2.  模拟三组实验观测：`p_A_vs_C`, `p_B_vs_C`, `p_A_vs_B`。例如，`p_A_vs_C = exp_map(sA.v_eigen + sC.v_eigen)` 这里的舞台是C。
        3.  执行三次测量：
            -   `s_A_vs_C = measure_source(p_total=p_A_vs_C, p_stage=p_C)`
            -   `s_B_vs_C = measure_source(p_total=p_B_vs_C, p_stage=p_C)`
            -   `s_A_vs_B = measure_source(p_total=p_A_vs_B, p_stage=p_B)`
    -   **验证**: 根据理论，替换效应应该是线性可加的。因此，`s_A_vs_B.v_eigen` 应该约等于 `s_A_vs_C.v_eigen - s_B_vs_C.v_eigen`。这可以通过 `np.allclose()` 来验证。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 此模块是纯 NumPy 实现，不直接涉及 JAX 后端。其对 `log_map_from_origin` 的依赖意味着它的行为一致性已经由 `cgd/geometry` 模块的后端一致性测试间接保证。

---
### 文件名

`cgd/measure/calibration.py`

### 核心功能详述

该模块提供了 `AlphaFitter` 类，其核心使命是**校准宇宙的物理常数 `alpha`**。它通过比较理论模型在不同 `alpha` 值下的预测与真实的“竞争实验”观测数据，来寻找能够最好地解释观测结果的最优 `alpha` 值。

-   **核心逻辑**:
    -   **目标函数**: `_objective_function_single` 是其核心。对于一个给定的 `alpha` 候选值，它会遍历所有的竞争实验数据。在每个实验中，它会：
        1.  创建一个具有当前 `alpha` 候选值的 `Universe`。
        2.  调用 `universe.find_equilibria` 来预测理论上的平衡点。
        3.  计算所有预测出的平衡点与真实的 `p_observed` 之间的 `distance_FR`，并取最小值。
        4.  将这个最小距离的平方作为该实验的“误差”。
    -   所有实验的误差被累加起来，得到该 `alpha` 候选值的总误差。
-   **拟合策略**: `fit` 方法采用了一种稳健的两阶段优化策略来最小化总误差：
    1.  **全局网格搜索**: 在用户定义的 `alpha_range` 内生成一系列 `alpha` 候选值，并使用 `joblib` **并行地**计算每个候选值的总误差。这一步的目的是粗略地找到误差最小的区域。
    2.  **局部精确优化**: 从网格搜索找到的最佳 `alpha` 值出发，使用 `scipy.minimize` (L-BFGS-B方法) 进行高精度的局部搜索，以找到最终的最优 `alpha` 值。
-   **双后端支持**: `AlphaFitter` 的设计是后端无关的。它通过在内部创建 `Universe` 时传入 `backend` 标志 (`'numpy'` 或 `'jax'`) 来无缝地利用任一后端进行计算，这使得用户可以轻松利用 JAX 后端进行高性能拟合。
-   **输入验证**: 在初始化时，它会检查所有传入的 `substitution` 类型的源是否具有相同的 `labels`，确保了比较的有效性。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.source.GravitationalSource`: 需要使用源的 `embed` 方法。
    -   `cgd.core.universe.Universe`: **强依赖**，是进行理论预测的核心。
    -   `cgd.geometry.simplex.distance_FR`: 用于计算预测与观测之间的误差。
-   **外部依赖**:
    -   `numpy`: 核心数值计算。
    -   `scipy.optimize.minimize`: 用于局部精确优化。
    -   `joblib`: 用于并行化网格搜索。
    -   `os`, `warnings`: 标准库。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   我们将**正向**模拟一个已知 `alpha` 值的宇宙来生成测试数据。
    1.  定义一个“真实”的 `alpha_true = 1.2`。
    2.  创建一个 `K=3` 的 `Universe`，`alpha=alpha_true`。
    3.  创建一套 `source_map`，包含 `s_stage`, `sA`, `sB`。
    4.  模拟一个竞争实验：调用 `universe.find_equilibria([s_stage, sA, sB])`，将其返回的第一个解作为模拟的“观测数据” `p_observed_sim`。
    5.  构造 `competition_data` 字典，指向 `p_observed_sim`。
    6.  将 `source_map` 和 `competition_data` 作为 `AlphaFitter` 的输入。

---

#### A. 正常功能测试 (Happy Path)

1.  **最优 Alpha 拟合**:
    -   **Case**: 使用上述策略生成测试数据，然后调用 `fitter.fit(alpha_range=(0.5, 2.0))`。
    -   **验证**: 拟合出的 `alpha_optimal` 应该非常接近我们预设的 `alpha_true` (1.2)。
2.  **后端切换**:
    -   **Case**: 同样的数据，分别使用 `backend='numpy'` 和 `backend='jax'` 初始化 `AlphaFitter` 并进行拟合。
    -   **验证**: 两个后端拟合出的 `alpha_optimal` 应该高度一致。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`alpha_true = 0`**:
    -   **Case**: 使用 `alpha_true = 0.0` 生成模拟观测数据。此时，观测结果应与LESA预测完全一致。
    -   **验证**: `AlphaFitter` 拟合出的最优 `alpha` 应非常接近于 0。
2.  **数据完美匹配LESA (alpha=0)**:
    -   **Case**: 即使 `alpha_true=1.2`，但如果手动将 `p_observed` 设置为LESA的预测结果。
    -   **验证**: `AlphaFitter` 应该返回一个接近于0的最优 `alpha`，因为它会发现 `alpha=0` 能完美解释数据。
3.  **单点网格搜索**:
    -   **Case**: 调用 `fit` 时设置 `grid_points=1`。
    -   **验证**: 程序应能正常运行，不应因无法计算搜索宽度而失败。

#### C. 异常与验证测试 (Error & Validation)

1.  **初始化 `ValueError` - 坐标系不匹配**:
    -   **Case**: `source_map` 中包含两个 `substitution` 类型的源，但它们的 `labels` 不同。
    -   **验证**: 初始化 `AlphaFitter` 时应抛出 `ValueError`。
2.  **初始化 `ValueError` - 数据键缺失**:
    -   **Case**: `competition_data` 中的某个实验条目缺少 `'labels'` 或 `'sources'` 键。
    -   **验证**: 初始化时应抛出 `ValueError`。
3.  **拟合失败 `RuntimeError`**:
    -   **Case**: （需要 `mock` `universe.find_equilibria` 使其总是返回空列表）模拟所有 `alpha` 值都无法找到解的情况。
    -   **验证**: `fit` 方法应抛出 `RuntimeError`，提示所有网格点误差均为无穷大。
4.  **局部优化失败**:
    -   **Case**: (需要 `mock` `scipy.minimize` 使其 `success` 属性为 `False`)。
    -   **验证**: 函数应能正常回退，返回网格搜索的结果，并触发一个 `UserWarning`。

#### D. 理论自洽性测试 (Theoretical Consistency)

-   **说明**: `AlphaFitter` 本身就是对 `cgd` 理论自洽性（即理论模型能以特定 `alpha` 值解释实验数据）的宏观检验。A.1 中的测试是其核心的自洽性验证。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 这是本模块的一个**关键集成测试**。
1.  **`_objective_function_single` 一致性**:
    -   **Case**: 对于一个固定的 `alpha_candidate`，分别使用 NumPy 和 JAX 后端的 `AlphaFitter` 实例调用 `_objective_function_single`。
    -   **验证**: 两个后端计算出的总误差应高度一致。这间接验证了两个后端的 `EquilibriumFinder` 在这个特定场景下找到了相同（或几何距离相同）的解。
2.  **`fit` 结果一致性**:
    -   **Case**: A.2 的测试用例。
    -   **验证**: 确保两个后端不仅单点误差一致，而且最终的全局+局部优化流程也能收敛到同一个最优 `alpha` 值。

---
### 文件名

`cgd/measure/discovery.py`

### 核心功能详述

该模块提供了 `ProbeFitter` 类，这是 `cgd` 理论实现**自我修正和演化**的核心引擎。当理论模型（即已知的 `source_map` 和 `alpha`）的预测结果 `p_predicted` 与真实的实验观测 `p_observed` 之间出现无法解释的偏差时，`ProbeFitter` 的任务就是从这个偏差中“发现”一个或多个新的、未知的引力源（即“探针”）。

-   **核心逻辑**:
    1.  **输入**: 接收理论预测点 `p_predicted` 和真实观测点 `p_observed`。
    2.  **解码到效应空间**: 它首先使用 `log_map_from_origin` 将这两个点都转换到线性的“效应空间”，得到 `v_predicted` 和 `v_observed`。
    3.  **定义目标函数**: `_objective_function` 是其核心。它以一个候选的探针效应向量 `v_probe` 为输入，然后计算 `exp_map_from_origin(v_predicted + v_probe)` 与 `p_observed` 之间的几何距离。其优化的目标就是找到一个 `v_probe`，使得加入这个探针后的新预测能完美匹配观测结果（即距离为零）。
    4.  **高质量初始猜测**: 在 `fit` 方法中，它使用残差向量 `v_observed - v_predicted` 作为 `scipy.minimize` 优化的初始猜测点。这是一个极佳的起点，因为它正是 `alpha=0` 时的精确解。
    5.  **拟合与封装**: `fit` 方法调用 `scipy.minimize` 来找到最优的 `v_probe_optimal`。然后，它将这个向量封装成一个信息完备的 `GravitationalSource` 对象（类型为 `'absolute'`，因为探针解释的是绝对偏差）并返回。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.source.GravitationalSource`: 用于封装和返回最终的探针。
    -   `cgd.geometry.simplex.distance_FR`: 作为优化的目标函数（误差函数）。
    -   `cgd.geometry.transformations`: `log_map_from_origin` 和 `exp_map_from_origin` 是其算法的核心。
-   **外部依赖**:
    -   `numpy`: 核心数值计算。
    -   `scipy.optimize.minimize`: 用于执行拟合。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   我们将模拟一个存在未知源的场景来生成测试数据。
    1.  创建两个已知源 `s_known_1`, `s_known_2`，它们代表我们当前的“理论模型”。
    2.  创建一个“未知”源 `s_unknown`，这就是我们要发现的“探针”。
    3.  计算理论预测：`v_predicted = s_known_1.v_eigen + s_known_2.v_eigen`，`p_predicted = exp_map(v_predicted)`。
    4.  计算“真实观测”：`v_observed = v_predicted + s_unknown.v_eigen`，`p_observed = exp_map(v_observed)`。
    5.  将 `p_predicted` 和 `p_observed` 作为 `ProbeFitter` 的输入。

---

#### A. 正常功能测试 (Happy Path)

1.  **探针发现**:
    -   **Case**: 使用上述策略生成 `p_predicted` 和 `p_observed`，然后调用 `fitter.fit()`。
    -   **验证**: 拟合出的探针 `probe_found` 的 `v_eigen` 应该与我们预设的 `s_unknown.v_eigen` 在数值上高度一致 (`np.allclose`)。

#### B. 边界条件测试 (Boundary Conditions)

1.  **无偏差**:
    -   **Case**: 输入的 `p_predicted` 和 `p_observed` 完全相同。
    -   **验证**: `fit()` 方法应能成功运行，并返回一个 `v_eigen` 为零向量的探针。
2.  **使用 `alpha > 0` 生成的数据**:
    -   **Case**:
        1.  用一个 `alpha > 0` 的宇宙来生成 `p_predicted` (来自已知源) 和 `p_observed` (来自已知源+未知源)。
        2.  将这两个点输入给 `ProbeFitter`。
    -   **验证**: `ProbeFitter` 的拟合是基于 `alpha=0` 的LESA假设的。因此，当真实数据由 `alpha > 0` 的非线性过程产生时，拟合出的探针 `v_probe` **不**会精确等于 `s_unknown.v_eigen`。测试应验证函数能成功返回一个结果，但不对其精确性做要求。这验证了 `ProbeFitter` 是 `alpha=0` 理论框架下的一阶修正工具。

#### C. 异常与验证测试 (Error & Validation)

1.  **维度不匹配**:
    -   **Case**: 初始化 `ProbeFitter` 时，传入维度不匹配的 `p_predicted` 和 `p_observed`。
    -   **验证**: 应抛出 `ValueError`。
2.  **拟合失败**:
    -   **Case**: (需要 `mock` `scipy.minimize` 使其 `success` 属性为 `False`)。
    -   **验证**: `fit()` 方法应抛出 `RuntimeError`。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **发现与净化的关系**:
    -   **Case**:
        1.  `p_predicted` 设为宇宙中心点（代表 `v_predicted=0`）。
        2.  `p_observed` 设为 `p_total`。
        3.  调用 `probe = ProbeFitter(p_predicted, p_observed).fit()`。
        4.  调用 `source = measure_source(p_total=p_total, p_stage=p_predicted)`。
    -   **验证**: `probe.v_eigen` 应与 `source.v_eigen` 完全相同。这验证了当没有先验理论时（`v_predicted=0`），“发现”过程就等同于一次绝对的“测量”。
2.  **探针的校正作用**:
    -   **Case**:
        1.  使用 A.1 的测试用例，得到 `probe_found`。
        2.  计算修正后的预测：`v_corrected = v_predicted + probe_found.v_eigen`，`p_corrected = exp_map(v_corrected)`。
    -   **验证**: `p_corrected` 应与 `p_observed` 高度一致 (`np.allclose`)。这验证了发现的探针确实能够完美地解释理论与观测之间的偏差。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: 此模块是纯 NumPy 实现，不直接涉及 JAX 后端。其对几何模块的依赖意味着其行为一致性已经由 `cgd/geometry` 模块的后端一致性测试间接保证。

---
### 文件名

`cgd/measure/refinement.py`

### 核心功能详述

该模块提供了 `VectorInverter` 类，其核心任务是在一个**已知 `alpha` 值**的宇宙中，对通过几何净化 (`measure_source`) 得到的 `v_eigen` 进行**高精度精炼**。这个过程是必要的，因为 `measure_source` 是在 `alpha=0` 的线性假设下进行的，当 `alpha > 0` 时，源之间的非线性相互作用会使这个初步估计产生偏差。`VectorInverter` 的作用就是修正这个偏差。

-   **核心逻辑**:
    -   **物理前提**: 理论上，实验观测到的平衡点 `p_total_obs` 是一个稳定的吸引子，这意味着在该点，由所有源（已知的 `stage_source` 和我们正在精炼的 `target_source`）产生的合力 `F_net` 必须为零。
    -   **输入**:
        -   `p_total_obs`: 真实的实验观测平衡点。
        -   `stage_source`: 已知的背景源。
        -   `universe`: 一个已经用**固定的 `alpha` 值**初始化的宇宙。
        -   `initial_guess_v_eigen`: 目标源 `v_eigen` 的一个初步估计（通常来自 `measure_source`）。
    -   **目标函数**: `_objective_function_fast` 的目标是找到一个 `v_eigen_candidate`，使得由 `stage_source` 和这个候选源共同作用时，在 `p_total_obs` 点产生的合力的模长平方 (`||F_net||^2`) 最小化（理想情况下为零）。
    -   **优化与输出**: `refine` 方法使用 `scipy.minimize` 来执行这个优化，并返回一个包含了**精炼后 `v_eigen`** 的 `GravitationalSource` 对象。这个新的 `v_eigen` 是在该 `alpha` 宇宙中与观测数据最一致的精确解。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.source.GravitationalSource`: 用于创建候选源和返回最终结果。
    -   `cgd.core.universe.Universe`: 定义了优化的物理环境（主要是 `alpha`）。
    -   `cgd.dynamics.force.calculate_F_net`: **强依赖**，是其目标函数的核心。
-   **外部依赖**:
    -   `numpy`: 核心数值计算。
    -   `scipy.optimize.minimize`: 用于执行优化。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   我们将正向模拟一个 `alpha > 0` 的宇宙来生成测试数据。
    1.  定义一个“真实”的 `alpha_true = 1.5`。
    2.  创建两个“真实”的源 `s_stage_true` 和 `s_target_true`。
    3.  创建一个 `alpha=alpha_true` 的 `Universe`。
    4.  找到这个宇宙中的平衡点 `p_obs_sim = universe.find_equilibria([s_stage_true, s_target_true])[0]`。这个点将作为模拟的“真实观测”。
    5.  进行一次初步的几何净化测量，得到一个有偏差的估计：`s_target_initial = measure_source(p_total=p_obs_sim, p_stage=exp_map(s_stage_true.v_eigen))`。
    6.  将 `p_obs_sim`, `s_stage_true`, 和 `s_target_initial.v_eigen` 作为 `VectorInverter` 的输入。

---

#### A. 正常功能测试 (Happy Path)

1.  **向量精炼**:
    -   **Case**: 使用上述策略生成测试数据，然后调用 `inverter.refine(s_target_initial.v_eigen)`。
    -   **验证**: 返回的 `s_target_refined` 的 `v_eigen` 应该比 `s_target_initial.v_eigen` 更接近“真实”的 `s_target_true.v_eigen`。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`alpha = 0`**:
    -   **Case**: `universe` 的 `alpha` 设置为 `0.0`。使用 `measure_source` 的结果作为 `initial_guess_v_eigen`。
    -   **验证**: 当 `alpha=0` 时，没有非线性效应需要修正。因此，精炼后的 `v_eigen` 应该与输入的 `initial_guess_v_eigen` 完全相同，优化器应该在第一步就收敛。
2.  **初始猜测已是精确解**:
    -   **Case**: 直接使用“真实”的 `s_target_true.v_eigen` 作为 `initial_guess_v_eigen`。
    -   **验证**: 优化器应该立即收敛，返回的 `v_eigen` 与输入的 `v_eigen` 相同。

#### C. 异常与验证测试 (Error & Validation)

1.  **优化失败**:
    -   **Case**: (需要 `mock` `scipy.minimize` 使其 `success` 属性为 `False`)。
    -   **验证**: 方法应能正常返回一个结果（即优化器停止时的值），并打印出警告信息。

#### D. 理论自洽性测试 (Theoretical Consistency)

1.  **精炼结果的自洽性 (核心测试)**:
    -   **Case**:
        1.  使用 A.1 的测试用例，得到精炼后的源 `s_target_refined`。
        2.  使用这个精炼后的源和 `s_stage_true`，在同一个 `alpha=alpha_true` 的宇宙中重新寻找平衡点：`p_predicted_refined = universe.find_equilibria([s_stage_true, s_target_refined])[0]`。
    -   **验证**: `p_predicted_refined` 应该与我们最初的模拟观测 `p_obs_sim` 在数值上高度一致 (`np.allclose`)。这验证了精炼过程确实找到了能够在该 `alpha` 宇宙中重现观测结果的正确 `v_eigen`。

#### E. JAX与NumPy后端一致性测试 (Backend Consistency)

-   **说明**: `VectorInverter` 依赖于 `calculate_F_net`。虽然 `VectorInverter` 本身没有 JAX 版本，但可以通过向其传入一个使用 `'jax'` 后端的 `Universe` 来间接测试。
1.  **通过 `Universe` 后端切换进行测试**:
    -   **Case**: (需要修改 `VectorInverter` 使其能够使用 JAX 后端的 `calculate_F_net_jax`)。或者，创建一个手动的测试，其目标函数直接调用 JAX 版本的 `calculate_F_net_jax`。
    -   **验证**: 由于 JAX 版本的梯度是精确的，而 NumPy 版本是近似的，两个后端优化出的 `v_refined` 可能会有微小差异。测试应验证这两个 `v_refined` 在合理的容差下是相近的。更重要的是，它们都应该满足 D.1 的自洽性测试，即都能重现观测结果。

---
## 模块: `cgd.utils`, `cgd.visualize`, `cgd.workflows`

### 文件名

`cgd/utils/data_helpers.py`

### 核心功能详述

该模块提供了一系列高级辅助函数，旨在**简化从原始实验数据到 `cgd` 核心对象的转换过程**，充当了数据预处理层。这些函数极大地提升了库的易用性，让用户可以从常见的原始数据格式（如 `pandas.DataFrame` 或字典）直接开始分析。

-   **`process_counts_to_p_dict`**: 这是一个通用的数据清洗和转换函数。
    -   **功能**: 它接收多种格式的原始事件计数（counts）数据，并将其转换为一个规范化的概率分布字典 (`p_dict`)。
    -   **鲁棒性**: 它能自动处理多种棘手情况，例如总计数为零的组（返回均匀分布并发出警告）、通过添加伪计数来避免概率为零（增强数值稳定性）、以及从 `DataFrame` 中正确地按组提取和重排数据。
-   **`create_source_map_from_data`**: 这是一个更高级的封装函数，自动化了从原始数据到创建 `GravitationalSource` 对象的完整流程。
    -   **工作流**: 它首先调用 `process_counts_to_p_dict` 进行数据清洗，然后自动识别“舞台”（`stage_group_name`）和“刺激物”（`stim_group_prefix`）组，并成批地调用 `measure_source` 来执行几何净化。
    -   **逻辑正确性**: 它正确地将“舞台”源视为相对于宇宙真空（均匀分布）的 `'absolute'` 效应进行测量，并将“刺激物”源视为相对于“舞台”的效应进行测量。
    -   **输出**: 返回一个 `source_map` 字典，可以直接用于后续的 `AlphaFitter` 或 `Universe` 模拟。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.source.GravitationalSource`: `create_source_map_from_data` 创建此类型的对象。
    -   `cgd.measure.purification.measure_source`: `create_source_map_from_data` 的核心依赖。
-   **外部依赖**:
    -   `numpy`: 用于数值计算。
    -   `pandas`: 用于处理 `DataFrame` 输入。
    -   `warnings`: 用于处理异常数据情况。

### 初步测试计划 (Test Plan)

**测试数据生成策略**:
-   创建一个 `K=3` 的 `event_labels = ['A', 'B', 'C']`。
-   创建一个 `pandas.DataFrame` 和一个字典，包含相同的计数数据，用于测试两种输入格式。
    -   **DataFrame**:
        ```
        group    event
        Stage    A
        Stage    A
        Stage    B
        Stim_X   C
        Stim_X   C
        Stim_X   C
        Stim_Y   A
        Stim_Y   B
        ```
    -   **字典**:
        ```
        {
          "Stage":  [2, 1, 0],
          "Stim_X": [0, 0, 3],
          "Stim_Y": [1, 1, 0]
        }
        ```

---

#### A. 正常功能测试 (Happy Path)

1.  **`process_counts_to_p_dict` - 字典输入**:
    -   **Case**: 使用测试字典作为输入。
    -   **验证**: 返回的 `p_dict` 应包含三个键，每个键对应的值都是一个归一化（且包含伪计数）的 `numpy` 数组。例如，`p_dict['Stage']` 应约等于 `[2/3, 1/3, 0]`。
2.  **`process_counts_to_p_dict` - DataFrame 输入**:
    -   **Case**: 使用测试 `DataFrame` 作为输入。
    -   **验证**: 返回的 `p_dict` 应与字典输入的结果完全相同。
3.  **`create_source_map_from_data`**:
    -   **Case**: 使用任一测试数据源调用此函数。
    -   **验证**:
        -   返回的 `source_map` 应包含三个 `GravitationalSource` 对象：`'Stage'`, `'X'`, `'Y'`。
        -   `source_map['Stage']` 的 `v_type` 应为 `'absolute'`。
        -   `source_map['X']` 和 `source_map['Y']` 的 `v_type` 应为 `'substitution'` (默认值)。

#### B. 边界条件测试 (Boundary Conditions)

1.  **`process_counts_to_p_dict` - 零计数**:
    -   **Case**: 数据中某个组的总计数为0。
    -   **验证**: 该组对应的概率向量应为均匀分布 `[1/3, 1/3, 1/3]`，并应触发一个 `UserWarning`。
2.  **`create_source_map_from_data` - 无刺激物组**:
    -   **Case**: 数据中只有 `Stage` 组，没有以 `Stim_` 开头的组。
    -   **验证**: 函数应能正常运行，返回只包含 `'Stage'` 源的 `source_map`，并触发一个 `UserWarning`。

#### C. 异常与验证测试 (Error & Validation)

1.  **`process_counts_to_p_dict` - `TypeError`**:
    -   **Case**: 传入一个不支持的数据类型（例如，列表）。
    -   **验证**: 抛出 `TypeError`。
2.  **`process_counts_to_p_dict` - `ValueError` (DataFrame)**:
    -   **Case**: 使用 `DataFrame` 输入但没有提供 `group_col`。
    -   **验证**: 抛出 `ValueError`。
3.  **`process_counts_to_p_dict` - `ValueError` (字典)**:
    -   **Case**: 字典中某一组的计数值数量与 `event_labels` 长度不匹配。
    -   **验证**: 抛出 `ValueError`。
4.  **`create_source_map_from_data` - `ValueError`**:
    -   **Case**: 数据中缺少 `stage_group_name` 指定的组。
    -   **验证**: 抛出 `ValueError`。

#### D/E. 理论自洽性/后端一致性测试

-   **说明**: 此模块是纯粹的数据预处理工具，不涉及深刻的理论假设或后端计算。其正确性通过与下游模块（如 `AlphaFitter`）的集成测试来间接验证。

---
### 文件名

`cgd/visualize/atlas.py`

### 核心功能详述

该模块提供了 `AtlasPlotter` 类，这是一个专门用于**可视化宇宙分析（alpha扫描）结果**的工具。它接收一个由 `run_universe_analysis` 等工作流函数生成的、包含多种分析指标随 `alpha` 值变化的数据字典，并提供了绘制三种核心分析图表的方法。

-   **`plot_phase_diagram`**: 绘制宇宙相图，展示吸引子数量随 `alpha` 变化的阶梯状关系。
-   **`plot_lesa_deviation`**: 绘制LESA偏差图，展示真实平衡点与线性预测的偏差随 `alpha` 的变化。
-   **`plot_drift_trajectory`**: 绘制平衡点漂移轨迹的PCA二维投影，并用颜色映射 `alpha` 值。

该类的主要作用是将复杂的分析数据转化为直观、可解释的科学图表。

### 依赖关系分析

-   **内部依赖**: 无。
-   **外部依赖**:
    -   `numpy`: 用于数据处理。
    -   `matplotlib.pyplot`: 用于绘图。

### 初步测试计划 (Test Plan)

-   **测试策略说明**:
    -   GUI和绘图代码的自动化单元测试通常投资回报率较低。主要测试策略是**“冒烟测试” (Smoke Testing)**。
    -   “冒烟测试”的目标是验证绘图函数在给定有效数据时能否成功执行并生成图像，而不会抛出异常。
    -   图像的正确性通常通过**视觉回归测试 (Visual Regression Testing)** 来保证，这需要专门的框架（如 `pytest-mpl`）。

---

#### A. 正常功能测试 (Happy Path) - 冒烟测试

1.  **冒烟测试所有绘图方法**:
    -   **Case**:
        1.  创建一个包含所有必需键 (`'alphas'`, `'n_attractors'`, etc.) 的模拟 `scan_results` 字典。数据应具有正确的形状和类型。
        2.  创建一个 `matplotlib` 的 `Figure` 和 `Axes` 对象。
        3.  初始化 `plotter = AtlasPlotter(scan_results)`。
        4.  依次调用 `plotter.plot_phase_diagram(ax)`, `plotter.plot_lesa_deviation(ax)`, `plotter.plot_drift_trajectory(ax)`。
    -   **验证**: 每一个绘图调用都应能成功执行，不抛出任何异常。

#### B/C. 边界条件/异常测试

1.  **测试数据格式错误**:
    -   **Case**: 传入一个缺少键（例如，缺少 `'n_attractors'`）的 `scan_results` 字典。
    -   **验证**: 调用相应的绘图方法（如 `plot_phase_diagram`）时，应抛出 `KeyError`。
2.  **测试数据维度错误**:
    -   **Case**: `alphas` 是一维数组，但 `drift_trajectory_pca` 是一个一维数组（而它应该是 N x 2 的二维数组）。
    -   **验证**: `plot_drift_trajectory` 在尝试索引 `[:, 0]` 时应抛出 `IndexError`。

---
### 文件名

`cgd/visualize/fingerprints.py`

### 核心功能详述

该模块提供了 `FingerprintPlotter` 类，这是一个专门用于将 `cgd` 的核心数据向量（`p` 和 `v_eigen`）可视化为直观的**雷达图**的工具。

-   **`plot_behavior_fingerprint`**: 用于绘制“行为指纹”，它将一个概率向量 `p` 在极坐标系上展示出来，每个轴代表一个事件的概率。
-   **`plot_effect_fingerprint`**: 用于绘制“效应指纹”，它将一个效应向量 `v_eigen` 在一个以“赤道”（零效应）为基准的极坐标系上展示出来，通过向外（正效应）或向内（负效应）的填充来直观表示源的作用模式。
-   **核心逻辑**: 该类在初始化时接收 `labels` 并计算好雷达图的各个轴的角度。绘图方法负责将输入的数据向量正确地映射到这些轴上，并使用 `matplotlib` 的极坐标绘图功能来渲染图像。

### 依赖关系分析

-   **内部依赖**: 无。
-   **外部依赖**:
    -   `numpy`: 用于数据处理和角度计算。
    -   `matplotlib.pyplot`: 用于绘图。

### 初步测试计划 (Test Plan)

-   **测试策略说明**: 与 `AtlasPlotter` 类似，采用**冒烟测试**作为主要的自动化测试策略。

---

#### A. 正常功能测试 (Happy Path) - 冒烟测试

1.  **冒烟测试 `plot_behavior_fingerprint`**:
    -   **Case**:
        1.  创建一个 `p_vector = np.array([0.5, 0.3, 0.2])`。
        2.  创建一个 `matplotlib` 的极坐标 `Axes` 对象 (`subplot_kw={'projection': 'polar'}`)。
        3.  初始化 `plotter = FingerprintPlotter(labels=['A', 'B', 'C'])`。
        4.  调用 `plotter.plot_behavior_fingerprint(p_vector, ax)`。
    -   **验证**: 绘图调用成功执行，不抛出异常。
2.  **冒烟测试 `plot_effect_fingerprint`**:
    -   **Case**:
        1.  创建一个 `v_eigen = np.array([0.5, -0.3, -0.2])`。
        2.  使用与上面相同的 `plotter` 和 `ax`。
        3.  调用 `plotter.plot_effect_fingerprint(v_eigen, ax)`。
    -   **验证**: 绘图调用成功执行，不抛出异常。

#### B/C. 边界条件/异常测试

1.  **测试数据维度不匹配**:
    -   **Case**: `plotter` 使用3个 `labels` 初始化，但传入一个4维的 `p_vector`。
    -   **验证**: `plot_behavior_fingerprint` 应抛出 `ValueError`。
    -   **Case**: `plotter` 使用3个 `labels` 初始化，但传入一个4维的 `v_eigen`。
    -   **验证**: `plot_effect_fingerprint` 应抛出 `ValueError`。
2.  **测试零向量输入**:
    -   **Case**: 向 `plot_effect_fingerprint` 传入一个零向量 `v_eigen = np.zeros(3)`。
    -   **验证**: 函数应能成功执行（冒烟测试），正确处理 `max_abs_val == 0` 的情况，不产生除以零的错误。

---
### 文件名

`cgd/visualize/landscape.py`

### 核心功能详述

该模块提供了 `LandscapePlotter` 类，这是一个专门为 **K=3 的宇宙** 设计的可视化工具。它使用 `python-ternary` 库将三维的概率单纯形投影到一个二维的**三角图 (Terynary Plot)** 上，从而直观地展示系统的动力学特性。

-   **`plot_potential_surface`**: 绘制势能地貌的热力图。它会在三角图的每一个点上调用 `potential_gaussian` 计算势能，并用颜色深浅表示势能高低。
-   **`plot_force_field`**: 绘制合力场流线图。它会在三角图的网格点上调用 `calculate_F_net` 计算合力，并将力从单纯形坐标转换到 `ternary` 库的绘图坐标系，最终以流线的形式展示力的方向和大小。
-   **核心逻辑**: 关键在于 `potential_wrapper` 和 `force_wrapper` 这两个内部函数，它们是 `cgd` 库的坐标系统（`p` 向量和 `F` 向量）与 `python-ternary` 库的内部绘图坐标系统之间的桥梁。

### 依赖关系分析

-   **内部依赖**:
    -   `cgd.core.universe.Universe`, `cgd.core.source.GravitationalSource`: 用于定义绘图的物理环境。
    -   `cgd.dynamics.potential.potential_gaussian`: 用于绘制势能地貌。
    -   `cgd.dynamics.force.calculate_F_net`: 用于绘制合力场。
-   **外部依赖**:
    -   `numpy`: 用于数值计算。
    -   `matplotlib.pyplot`: 用于绘图。
    -   `ternary`: **强依赖**，是进行三角图绘制的基础库。

### 初步测试计划 (Test Plan)

-   **测试策略说明**: 采用**冒烟测试**。

---

#### A. 正常功能测试 (Happy Path) - 冒烟测试

1.  **冒烟测试完整绘图流程**:
    -   **Case**:
        1.  创建一个 `K=3` 的 `Universe` 和一个 `sources` 列表。
        2.  创建一个 `matplotlib` 的 `Figure` 和 `Axes` 对象。
        3.  初始化 `plotter = LandscapePlotter(universe, sources)`。
        4.  调用 `tax = plotter.plot_potential_surface(ax)`。
        5.  调用 `plotter.plot_force_field(tax)`。
    -   **验证**: 所有调用都成功执行，不抛出异常。

#### B/C. 边界条件/异常测试

1.  **测试维度检查**:
    -   **Case**: 尝试使用一个 `K=4` 的 `Universe` 来初始化 `LandscapePlotter`。
    -   **验证**: 构造函数应抛出 `ValueError`。

---
### 文件名

`cgd/workflows/analysis.py`

### 核心功能详述

该模块是 `cgd` 库的**顶层用户接口**，提供了两个高级的端到端工作流函数，将库的多个核心功能（数据处理、测量、校准、模拟、诊断）串联起来，极大地简化了用户的标准分析流程。

-   **`run_measurement_pipeline`**: 执行完整的“**理论构建**”工作流。
    -   **功能**: 从原始计数数据一步到位地生成 `source_map`，并可选地拟合最优 `alpha` 值。
    -   **集成**: 它在内部调用 `create_source_map_from_data` 来处理数据和执行几何净化，然后（如果需要）调用 `process_counts_to_p_dict` 和 `AlphaFitter` 来执行校准。

-   **`run_universe_analysis`**: 执行完整的“**理论验证与探索**”工作流。
    -   **功能**: 对一个给定的理论（`source_map` 和 `alpha`）进行全面的动力学分析、可视化和诊断。
    -   **集成**: 它在内部调用 `Universe` 来寻找平衡点，可选地调用 `LandscapePlotter` 进行 K=3 可视化，并在发现偏差时可选地调用 `ProbeFitter` 来尝试发现新源。它还负责处理 `source.embed`，确保所有源都与分析宇宙的坐标系兼容。

### 依赖关系分析

-   **内部依赖**:
    -   几乎依赖于 `cgd` 库的所有核心模块：`core`, `measure`, `utils`, `visualize`, `geometry`。它是将所有这些底层组件集成在一起的最高层。
-   **外部依赖**:
    -   `numpy`, `pandas`, `matplotlib.pyplot`。

### 初步测试计划 (Test Plan)

-   **测试策略说明**:
    -   这两个函数是高级集成点，因此它们的测试是**集成测试**，而不是单元测试。
    -   目标是验证函数能否正确地调用和协调其所有依赖项，并处理各种参数组合，而不是重复测试底层模块的算法正确性。
    -   测试将大量使用 **`unittest.mock`** 来**模拟**底层类的行为（如 `AlphaFitter.fit`, `Universe.find_equilibria`），从而可以隔离地、快速地测试工作流函数的逻辑流、参数传递和错误处理是否正确。

---

#### A. 正常功能测试 (Happy Path) - 集成测试

1.  **`run_measurement_pipeline` - 完整流程**:
    -   **Case**:
        1.  提供包含 "Stage", "Stim_", 和竞争组的测试数据。
        2.  `mock` `create_source_map_from_data` 使其返回一个预定义的 `source_map`。
        3.  `mock` `AlphaFitter.fit` 使其返回一个预定义的 `alpha_optimal = 1.1`。
        4.  调用 `run_measurement_pipeline`，并传入 `competition_groups`。
    -   **验证**:
        -   `create_source_map_from_data` 被以正确的参数调用。
        -   `AlphaFitter` 被以正确的 `source_map` 和 `competition_data` 初始化。
        -   `AlphaFitter.fit` 被调用。
        -   函数返回的 `source_map` 和 `alpha` 与 `mock` 对象返回的值一致。
2.  **`run_measurement_pipeline` - 仅测量**:
    -   **Case**: 调用函数时不传入 `competition_groups`。
    -   **验证**: `create_source_map_from_data` 被调用，但 `AlphaFitter` **不**被初始化或调用。返回的 `alpha` 值为 `None`。
3.  **`run_universe_analysis` - 完整流程**:
    -   **Case**:
        1.  提供一个 `source_map`, `alpha` 值, `p_observed` 等参数。
        2.  `mock` `Universe.find_equilibria` 使其返回一个预定义的平衡点列表。
        3.  `mock` `ProbeFitter.fit` 使其返回一个预定义的“探针”源。
        4.  调用 `run_universe_analysis`。
    -   **验证**:
        -   `Universe` 被以正确的 `K`, `alpha`, `labels` 初始化。
        -   `source.embed` 被正确调用。
        -   `Universe.find_equilibria` 被调用。
        -   由于偏差存在，`ProbeFitter` 被初始化并调用。

#### B/C. 边界条件/异常测试

1.  **`run_measurement_pipeline` - 竞争组数据缺失**:
    -   **Case**: `competition_groups` 字典中包含一个数据中不存在的组名。
    -   **验证**: 函数应在构造 `comp_data_for_fitter` 时抛出 `ValueError`。
2.  **`run_universe_analysis` - `source_map` 键错误**:
    -   **Case**: `source_names_for_analysis` 列表中包含一个 `source_map` 中不存在的源名称。
    -   **验证**: 函数应抛出 `KeyError`。
3.  **`run_universe_analysis` - 无平衡点**:
    -   **Case**: `mock` `Universe.find_equilibria` 使其返回一个空列表 `[]`。
    -   **验证**: 函数应能正常结束，不进行后续的绘图或探针拟合，并触发一个 `UserWarning`。
