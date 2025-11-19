# cgd/workflows/analysis.py
"""
cgd.workflows.analysis

封装端到端的高级分析工作流。
这是用户进行标准CGD分析的主要入口，内置了对双层理论体系
('absolute' vs 'substitution') 的支持和安全检查。
"""

import warnings
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 导入我们库的所有核心组件 ---
from ..core.source import GravitationalSource
from ..core.universe import Universe
from ..measure import AlphaFitter, ProbeFitter
# --- 核心修复：从 utils 导入我们需要的函数 ---
from ..utils import create_source_map_from_data, process_counts_to_p_dict
from ..visualize import FingerprintPlotter, LandscapePlotter, AtlasPlotter
from ..geometry import distance_FR


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
) -> Tuple[Dict[str, GravitationalSource], Optional[float]]:
    """
    执行完整的“靴袢式”测量工作流，从原始数据中测量引力源和可选的alpha。
    """
    print("--- 开始CGD测量工作流 ---")
    
    # --- 步骤1: 使用高级辅助函数一步到位地创建 source_map ---
    print(f"\n步骤1: 正在进行 '{measurement_type}' 类型的测量...")
    # create_source_map_from_data 内部已经处理了数据转换
    source_map = create_source_map_from_data(
        data=data,
        event_labels=event_labels,
        measurement_type=measurement_type,
        group_col=group_col,
        event_col=event_col,
        stage_group_name=stage_group_name,
        stim_group_prefix=stim_group_prefix
    )
    print("引力源测量完成。")
    for name, source in source_map.items():
        print(f"  - 已测量: {source}")

    # --- 步骤2: (如果需要) 进行alpha的一阶校准 ---
    alpha_optimal = None
    if competition_groups:
        print("\n步骤2: 正在进行一阶校准 (全局拟合 alpha)...")
        
        # --- 优化逻辑：只计算一次 p_map ---
        p_map = process_counts_to_p_dict(data, event_labels, group_col, event_col)
        
        comp_data_for_fitter = {}
        for group_name, stim_names_in_group in competition_groups.items():
            if group_name not in p_map:
                raise ValueError(f"在数据中找不到竞争实验组 '{group_name}'。")
            
            full_source_list = [stage_group_name] + stim_names_in_group
            
            comp_data_for_fitter[group_name] = {
                'K': len(event_labels),
                'sources': full_source_list,
                'p_observed': p_map[group_name],
                'labels': event_labels # <--- 为 AlphaFitter 提供标签信息
            }
        
        alpha_fitter = AlphaFitter(source_map, comp_data_for_fitter) 
        
        fitter_opts = alpha_fitter_options or {}
        alpha_optimal = alpha_fitter.fit(**fitter_opts)
        print(f"最优 alpha 拟合完成: α = {alpha_optimal:.4f}")
    
    print("\n--- CGD测量工作流完成 ---")
    return source_map, alpha_optimal


# --- run_universe_analysis 函数保持不变，因为它的重构是正确的 ---
def run_universe_analysis(
    source_map: Dict[str, GravitationalSource],
    alpha: float,
    analysis_labels: List[str],
    source_names_for_analysis: List[str],
    p_observed: Optional[np.ndarray] = None,
    validate_stability: bool = True,
    num_random_seeds: int = 30,
    plot: bool = True
):
    """
    对一个特定的宇宙进行完整的分析、可选的可视化和偏差诊断。
    """
    K = len(analysis_labels)
    print(f"\n--- 正在分析 K={K}, alpha={alpha:.2f} 的宇宙 (严格模式: {validate_stability}) ---")
    
    # Universe 现在也需要 labels
    universe = Universe(K, alpha, labels=tuple(analysis_labels))
    
    sources_for_analysis = []
    for name in source_names_for_analysis:
        if name not in source_map:
            raise KeyError(f"错误: 在 source_map 中找不到引力源 '{name}'。")
        
        source = source_map[name]
        embedded_source = source.embed(new_labels=analysis_labels)
        sources_for_analysis.append(embedded_source)
        
        if len(source.labels) != K:
            print(f"  - 已将源 '{name}' 从 K={len(source.labels)} ({source.labels}) 投影到 K={K} ({tuple(analysis_labels)})。")
            
    print("正在寻找平衡点...")
    equilibria = universe.find_equilibria(
        sources_for_analysis,
        validate_stability=validate_stability,
        num_random_seeds=num_random_seeds
    )
    
    if not equilibria:
        warnings.warn(f"在 alpha={alpha:.2f} 值下未找到稳定平衡点。")
        return

    print(f"找到了 {len(equilibria)} 个{'稳定' if validate_stability else '候选'}解。")
    p_predicted = equilibria[0]
    print(f"主要解: {np.round(p_predicted, 3)}")
    
    # (后续打印和可视化代码保持不变)
    if len(equilibria) > 1:
        print("其他解:")
        for i, p in enumerate(equilibria[1:]):
            print(f"  - 解 {i+2}: {np.round(p, 3)}")
    
    if plot and K == 3:
        try:
            import ternary
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            plotter = LandscapePlotter(universe, sources_for_analysis)
            tax = plotter.plot_potential_surface(ax, cmap='viridis')
            tax.scatter([tuple(p*100) for p in equilibria], marker='*', color='red', s=150, label='Predicted Equilibria', zorder=5)
            if p_observed is not None:
                tax.scatter([tuple(p_observed*100)], marker='o', color='cyan', s=100, label='Observation', zorder=5)
            tax.legend()
            tax.set_title(f"Landscape at alpha={alpha:.2f}", fontsize=14, pad=20)
            tax.left_axis_label(analysis_labels[1], offset=0.12)
            tax.right_axis_label(analysis_labels[0], offset=0.12)
            tax.bottom_axis_label(analysis_labels[2], offset=0.1)
            plt.show()
        except ImportError:
            warnings.warn("未安装 'python-ternary' 库，跳过 K=3 地貌图绘制。")
        except Exception as e:
            warnings.warn(f"绘制 K=3 地貌图时出错: {e}")

    # (偏差分析和探针发现代码保持不变)
    if p_observed is not None:
        if len(equilibria) > 1:
            distances = [distance_FR(p, p_observed) for p in equilibria]
            p_predicted = equilibria[np.argmin(distances)]
        
        print(f"\n将与最近的预测解进行比较...")
        print(f"  - 预测解: {np.round(p_predicted, 3)}")
        print(f"  - 真实观测: {np.round(p_observed, 3)}")
        deviation = distance_FR(p_predicted, p_observed)
        print(f"几何距离: {deviation:.4f}")

        if deviation > 0.01:
            print("\n发现显著偏差，正在尝试拟合探针...")
            try:
                probe_fitter = ProbeFitter(
                    p_predicted=p_predicted, 
                    p_observed=p_observed, 
                    event_labels=analysis_labels
                )
                probe = probe_fitter.fit(probe_name=f"Probe_for_{'_'.join(source_names_for_analysis)}")
                print(f"发现探针: {probe}")
            except Exception as e:
                print(f"探针拟合失败: {e}")