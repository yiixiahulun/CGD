# CGD 系统整合与架构分析报告 (v4.0)

**报告目的**: 本文档旨在从系统架构师的视角，对 `cgd` 代码库进行一次全面的整合层面审查。报告将超越单个模块的实现细节，聚焦于模块间的交互、核心数据流、架构的健康度以及系统层面的风险，并为未来的架构演进提供高层次的设计建议。

---

## 1. 核心数据流图 (Core Data Flow Diagram)

### 流程一: `run_measurement_pipeline` (理论构建)

此流程的核心任务是从原始数据中“测量”出引力源 (`GravitationalSource`) 对象，并可选地校准全局 `alpha` 值。数据流的核心是 **`GravitationalSource` 对象的创建与元数据的附加**。

```mermaid
graph TD
    subgraph "输入层"
        A[原始数据<br>(pd.DataFrame or Dict)]
        B[事件标签<br>(event_labels: List[str])]
        C[测量类型<br>(measurement_type: 'absolute'/'substitution')]
    end

    subgraph "cgd.workflows.analysis"
        D(run_measurement_pipeline)
    end

    subgraph "cgd.utils"
        E(create_source_map_from_data)
    end

    subgraph "cgd.measure"
        F(measure_source)
        G(AlphaFitter)
    end

    subgraph "输出层"
        H[引力源字典<br><b>source_map: Dict[str, GravitationalSource]</b>]
        I[最优Alpha值<br><b>alpha_optimal: float</b>]
    end

    A -- (counts) --> D
    B -- (labels) --> D
    C -- (v_type) --> D

    D -- (data, event_labels, measurement_type) --> E
    E -- (p_total, p_stage) --> F
    F --> H_Source["<b>GravitationalSource 对象</b><br>name: 'Stim_X'<br>v_type: 'substitution'<br>labels: ('A', 'B', 'C')<br>v_eigen: np.ndarray"]

    H_Source -- (source_map) --> D
    D -- (source_map) --> H

    subgraph "AlphaFitter 子流程"
        D -- (source_map, competition_data) --> G
        G -- (fit) --> I
    end

    classDef data fill:#f9f,stroke:#333,stroke-width:2px;
    class H_Source data;
```

**数据流解读**:
1.  `run_measurement_pipeline` 接收原始数据和关键的元数据 (`event_labels`, `measurement_type`)。
2.  它调用 `create_source_map_from_data`，该函数将原始数据转换为 `p_total` 和 `p_stage` 概率向量。
3.  这些向量被传递给 `measure_source`，最终实例化出 `GravitationalSource` 对象。
4.  **关键点**: `measurement_type` 和 `event_labels` 在此刻被永久地“烙印”在了 `GravitationalSource` 对象上，成为其 `v_type` 和 `labels` 属性。这些元数据从此跟随着对象，成为其理论身份的一部分。
5.  如果需要，这些携带了元数据的 `GravitationalSource` 对象和观测数据一起被送入 `AlphaFitter` 进行校准。

### 流程二: `run_universe_analysis` (理论应用与发现)

此流程的核心任务是使用已测量的引力源和 `alpha` 值来构建一个“宇宙”，并预测其行为。数据流的核心是 **`source.embed()` 的坐标系变换** 和 **`Universe` 作为后端的调度器**。

```mermaid
graph TD
    subgraph "输入层"
        J[引力源字典<br><b>source_map</b>]
        K[Alpha值<br>(alpha: float)]
        L[分析坐标系<br>(analysis_labels: List[str])]
        M[观测数据<br>(p_observed: np.ndarray)]
    end

    subgraph "cgd.workflows.analysis"
        N(run_universe_analysis)
    end

    subgraph "cgd.core"
        O(Universe)
        P(source.embed)
    end

    subgraph "cgd.dynamics / cgd.dynamics_jax"
        Q(EquilibriumFinder / EquilibriumFinderJax)
    end

    subgraph "cgd.measure"
        R(ProbeFitter)
    end

    subgraph "输出/发现"
        S[预测平衡点<br><b>p_predicted: np.ndarray</b>]
        T[探针源<br><b>Probe: GravitationalSource</b>]
    end

    J --> N
    K --> N
    L --> N
    M --> N

    N -- (K, alpha, labels) --> O
    N -- (source_map, analysis_labels) --> P
    P -- (embedded_source) --> N

    subgraph "Universe 内部"
      N -- (universe, embedded_sources) --> O_find["universe.find_equilibria(...)"]
      O_find -- (universe, sources) --> Q
    end

    Q -- (equilibria) --> O_find
    O_find -- (equilibria) --> S

    S -- (p_predicted) --> N
    N -- (p_predicted, p_observed) --> R
    R -- (fit) --> T

    classDef interaction fill:#ccf,stroke:#333,stroke-width:2px;
    class P, O_find interaction;
```
**数据流解读**:
1.  `run_universe_analysis` 接收 `source_map` 和定义目标宇宙的参数 (`alpha`, `analysis_labels`)。
2.  **关键交互**: 在将源送入 `Universe` 之前，工作流会为每个源调用 `source.embed(new_labels=analysis_labels)`。这是一个**显式的坐标系变换步骤**，它会返回一个新的、经过维度调整的 `GravitationalSource` 实例。
3.  `Universe` 对象被创建，并在内部根据 `backend` 参数动态绑定了求解器 (`EquilibriumFinder` 或 `EquilibriumFinderJax`)。
4.  `universe.find_equilibria` 被调用，它将嵌入后的源列表传递给内部绑定的求解器，最终计算出 `p_predicted`。
5.  如果提供了 `p_observed`，`ProbeFitter` 会利用预测与观测之间的偏差来发现新的“探针”引力源。

---

## 2. 模块耦合性与架构评估 (Module Coupling & Architectural Evaluation)

### 模块依赖关系

-   **健康的强耦合**:
    -   `dynamics` -> `geometry`: 物理引擎必然强依赖于底层的几何定义，这是健康且必要的耦合。
    -   `core` -> `dynamics`/`dynamics_jax`: `Universe` 作为核心调度器，需要知道可用的后端，这也是设计使然的健康耦合。
    -   `workflows` -> `core`, `measure`, `utils`: 工作流层作为顶层API，其职责就是编排和调用下层模块，耦合是其功能的一部分。
-   **健康的松耦合**:
    -   `measure` -> `core`: 测量模块需要与 `Universe` 和 `GravitationalSource` 交互，但它不关心 `dynamics` 的具体实现，耦合程度是合适的。
-   **可优化的耦合点**:
    -   **当前没有发现显著的、需要立即优化的不良耦合关系。** 模块之间的依赖关系遵循了良好的分层原则（高层依赖低层，核心模块不依赖应用模块）。

### 架构审查焦点1: 后端的隔离性

-   **评估**: `Universe` 类在隔离后端方面**基本成功，但存在一个明显的“泄露抽象”**。
-   **成功之处**:
    -   **策略模式**: `Universe` 的 `__post_init__` 方法通过动态导入和绑定 `_finder_class`，成功地实现了[策略设计模式](https://en.wikipedia.org/wiki/Strategy_pattern)。这使得 `universe.find_equilibria` 的调用代码完全不知道它最终会执行NumPy还是JAX的逻辑。
    -   **统一接口**: `EquilibriumFinder` 和 `EquilibriumFinderJax` 共享一个 `find` 方法，这为策略模式提供了必要的基础。
-   **泄露的抽象 (Leaky Abstraction)**:
    -   **问题**: `universe.find_equilibria` 方法通过 `**kwargs` 将参数直接传递给底层的 `find` 方法。
    -   **后果**: 这导致了调用者（如 `run_universe_analysis` 或 `AlphaFitter`）必须知道不同后端求解器的具体参数（例如，JAX后端需要 `learning_rate`，而NumPy后端不需要）。这破坏了 `Universe` 类的封装，后端实现的细节“泄露”到了上层的调用代码中。

### 架构审查焦点2: `SolverOptions` 的设计建议

为了修复上述的“泄露抽象”问题，并提供一个更清晰、可扩展的配置机制，我提出以下架构设计：

1.  **定义 `SolverOptions` 数据类**:
    -   创建一个 `cgd/dynamics/options.py` 文件，并定义一个 `dataclass`。
    -   这个 `dataclass` 将包含**所有后端**的可调参数，并提供合理的默认值。

    ```python
    # cgd/dynamics/options.py
    from dataclasses import dataclass

    @dataclass
    class SolverOptions:
        """Encapsulates all numerical options for equilibrium solvers."""
        # General options
        num_random_seeds: int = 20
        validate_stability: bool = True
        uniqueness_tolerance: float = 1e-4

        # NumPy-specific
        numpy_max_rounds: int = 50

        # JAX-specific
        jax_learning_rate: float = 0.05
        jax_cd_steps: int = 15
    ```

2.  **`SolverOptions` 的生命周期与传递路径**:
    -   **创建**: `SolverOptions` 对象应该在**最高层的API调用处**被创建，即在 `run_universe_analysis` 或 `AlphaFitter.fit` 的函数体内。这给了最终用户最大的控制权。
    -   **传递**:
        1.  `run_universe_analysis` 的签名应改为 `... (..., solver_options: Optional[SolverOptions] = None)`。
        2.  `Universe.find_equilibria` 的签名应改为 `... (self, ..., options: Optional[SolverOptions] = None)`。
        3.  `EquilibriumFinder.find` 和 `EquilibriumFinderJax.find` 的签名也应改为 `... (self, ..., options: SolverOptions)`。
    -   **消费**: 每个后端的 `find` 方法只从 `options` 对象中读取它自己关心的参数。

**架构优势**:
-   **封装性**: `Universe` 不再需要知道任何关于求解器参数的细节，它只负责传递一个不透明的 `options` 对象。
-   **类型安全**: 使用 `dataclass` 代替 `Dict[str, Any]` 提供了更好的静态分析和代码补全支持。
-   **可扩展性**: 未来增加新的后端或新的参数，只需要修改 `SolverOptions` 和对应的后端实现，而不需要修改中间的所有函数签名。

---

## 3. 整合层面的风险与假设分析 (Integration Risks & Assumption Analysis)

### 架构审查焦点3: 隐式假设

-   **假设**: `AlphaFitter` 依赖于一个名为 `'Stage'` 的源。
-   **分析**: 经过审查，`AlphaFitter` 类本身**并不**直接依赖于 `'Stage'` 源。它操作的是一个通用的 `source_map` 和 `competition_data` 结构。
-   **真正的来源**: 对 `'Stage'` 的依赖实际上存在于更高层的 `run_measurement_pipeline` 工作流中。这个工作流负责调用 `create_source_map_from_data`，而后者在 `measurement_type='substitution'` 模式下，其设计逻辑就是去创建和分离出一个 `'Stage'` 源。
-   **结论**: 这是一个**健康的关注点分离**。核心的 `AlphaFitter` 算法是通用的，而具体的实验协议（如需要一个 'Stage' 作为参照物）则被正确地封装在了工作流层。**未发现危险的隐式假设**。

### 架构审查焦点4: 理论一致性流 (`v_type` 和 `labels` 的传递)

-   **核心问题**: 架构本身是否能防止理论层面的错误，例如，将一个在K=4下测量的 `'substitution'` 源用于一个K=5的宇宙中？
-   **结论**: **能。** `cgd` 的架构拥有一套强大的、多层次的运行时检查机制，能非常有效地防止此类错误的发生。这些机制不是简单的文档警告，而是会主动抛出异常的硬性检查。

#### 多层防御体系 (Defense in Depth)

1.  **第一道防线 (`AlphaFitter`): 坐标系一致性检查**
    -   **位置**: `AlphaFitter.__init__`
    -   **机制**: 在初始化时，`AlphaFitter` 会遍历 `source_map` 中所有 `v_type='substitution'` 的源，并断言它们拥有**完全相同**的 `labels`。
    -   **作用**: 这在架构上保证了全局 `alpha` 拟合是在一个**统一的参照系**下进行的，从根本上杜绝了混合不同坐标系源的风险。这是一个至关重要的、前置的理论安全检查。

2.  **第二道防线 (`Universe`): 维度匹配检查**
    -   **位置**: `Universe._validate_sources`
    -   **机制**: 在任何核心计算（如 `find_equilibria`）发生前，`Universe` 会检查传入的每一个 `GravitationalSource` 对象的维度 (`len(source.v_eigen)`) 是否与宇宙自身的维度 `K` 完全匹配。
    -   **作用**: 这使得一个K=4的源**不可能**被意外地用在一个K=5的宇宙中。计算会立即失败，并给出清晰的错误信息。

3.  **指定的“转换门” (`GravitationalSource.embed`)**:
    -   **机制**: `embed` 方法是唯一允许改变源的 `labels` 和维度的地方。它返回一个**新的**源对象，而不是修改原始对象。
    -   **作用**: 由于第二道防线的存在，用户被**强制**必须调用 `.embed()` 来解决维度不匹配的问题。这个动作本身就成为了一个明确的、有意识的分析决策（“我确认要将这个源投影到新的坐标系中”），而不是一个可能被忽略的警告。

**总结**: `cgd` 的架构在理论一致性方面设计得非常出色。它不是依赖于用户的自觉，而是通过一系列连锁的、强制性的运行时检查，构建了一个“故障安全”(fail-safe)的系统。一个理论上不一致的操作（如混合坐标系或维度）会被架构主动拒绝，从而极大地降低了产生错误科学结论的风险。