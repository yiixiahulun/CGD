# cgd/utils/data_helpers.py
"""
提供一系列辅助函数，用于简化从不同数据格式到CGD核心对象的转换。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Literal
import warnings

# --- 从我们库的其他部分导入必要的对象和函数 ---
from ..core.source import GravitationalSource
from ..measure.purification import measure_source


def process_counts_to_p_dict(
    counts_data: Union[pd.DataFrame, Dict[str, List[int]]], 
    event_labels: List[str],
    group_col: Optional[str] = None,
    event_col: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    一个通用的辅助函数，将不同格式的原始计数数据转化为概率分布(p)字典。

    Args:
        counts_data (Union[pd.DataFrame, Dict[str, List[int]]]): 
            输入的实验数据。可以是DataFrame或计数字典。
        event_labels (List[str]): 
            事件的标签列表，定义了维度K和顺序。
        group_col (Optional[str]): 仅在 data 是 DataFrame 时使用。
        event_col (Optional[str]): 仅在 data 是 DataFrame 时使用。
        
    Returns:
        Dict[str, np.ndarray]: 
            一个字典，键是组名，值是对应的、经过清洗的概率向量。
    """
    p_dict = {}
    K = len(event_labels)
    
    if isinstance(counts_data, pd.DataFrame):
        if not group_col or not event_col:
            raise ValueError("使用DataFrame时，必须提供 group_col 和 event_col。")
        
        all_groups = counts_data[group_col].unique()
        for group in all_groups:
            subset = counts_data[counts_data[group_col] == group]
            if subset.empty:
                # 这是一个良性情况，可能数据中就是没有这个组，跳过即可
                warnings.warn(f"在DataFrame中找不到组 '{group}' 的任何数据，将跳过该组。")
                continue
            
            counts = subset[event_col].value_counts().reindex(event_labels, fill_value=0)
            p_vec = counts.values.astype(float)
            total = p_vec.sum()

            if total == 0:
                warnings.warn(f"组 '{group}' 的总计数为0，将使用均匀分布作为替代。")
                p_vec = np.full(K, 1.0 / K)
            else:
                # 添加伪计数1e-9以避免概率为0
                p_vec = (p_vec + 1e-9) / (total + K * 1e-9)
            
            p_dict[group] = p_vec

    elif isinstance(counts_data, dict):
        for group_name, counts in counts_data.items():
            if len(counts) != K:
                raise ValueError(f"组 '{group_name}' 的计数值数量 ({len(counts)}) "
                                 f"与事件标签数量 ({K}) 不匹配。")
            
            counts_arr = np.array(counts, dtype=float)
            total = np.sum(counts_arr)
            
            if total == 0:
                warnings.warn(f"组 '{group_name}' 的总计数为0，将使用均匀分布作为替代。")
                p_vec = np.full(K, 1.0 / K)
            else:
                p_vec = (counts_arr + 1e-9) / (total + K * 1e-9)

            p_dict[group_name] = p_vec
    else:
        raise TypeError(f"不支持的数据输入类型: {type(counts_data)}。")
        
    return p_dict


def create_source_map_from_data(
    data: Union[pd.DataFrame, Dict[str, List[int]]], 
    event_labels: List[str],
    measurement_type: Literal['absolute', 'substitution'] = 'substitution',
    group_col: Optional[str] = None, 
    event_col: Optional[str] = None,
    stage_group_name: str = 'Stage',
    stim_group_prefix: str = 'Stim_'
) -> Dict[str, GravitationalSource]:
    """
    一个高级辅助函数，从原始数据一步到位地创建引力源字典 (source_map)。
    
    这个函数封装了“数据处理 -> 概率转换 -> 几何净化 -> 对象创建”的完整流程。
    它是 `run_measurement_pipeline` 函数理想的预处理工具。

    Args:
        (参数文档保持不变)
        ...

    Returns:
        Dict[str, GravitationalSource]: 
            一个包含测量出的、信息完备的 GravitationalSource 对象的字典 (source_map)。
    """
    p_map = process_counts_to_p_dict(data, event_labels, group_col, event_col)
    K = len(event_labels)
    
    if stage_group_name not in p_map:
        raise ValueError(f"在数据中找不到舞台基线组 '{stage_group_name}'。")
    p_stage = p_map[stage_group_name]
    
    # 测量舞台偏好。舞台效应总是相对于绝对真空（均匀分布）而言的。
    p_vacuum_origin = np.full(K, 1.0/K)
    stage_source = measure_source(
        name=stage_group_name,
        p_total=p_stage,
        p_stage=p_vacuum_origin,
        event_labels=event_labels, # <--- 核心修复 1
        measurement_type='absolute'
    )
    source_map = {stage_group_name: stage_source}

    stim_groups = [g for g in p_map.keys() if g.startswith(stim_group_prefix)]
    if not stim_groups:
         warnings.warn(f"未找到任何单刺激实验组 (前缀: '{stim_group_prefix}')。")
    
    for group in stim_groups:
        stim_name = group.replace(stim_group_prefix, '')
        p_total = p_map[group]
        
        stim_source = measure_source(
            name=stim_name,
            p_total=p_total,
            p_stage=p_stage,
            event_labels=event_labels, # <--- 核心修复 2
            measurement_type=measurement_type
        )
        source_map[stim_name] = stim_source

    return source_map