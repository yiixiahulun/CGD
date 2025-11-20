from typing import List, Dict, Literal
import numpy as np
import pandas as pd
from cgd.core.source import GravitationalSource
from cgd.core.universe import Universe
from cgd.geometry.transformations import log_map_from_origin
def create_test_source(p: np.ndarray, name: str, v_type: Literal['absolute', 'substitution'], labels: List[str]) -> GravitationalSource:
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Input probability vector p must sum to 1.")
    v_eigen = log_map_from_origin(p)
    return GravitationalSource(name=name, v_type=v_type, labels=tuple(labels), v_eigen=v_eigen)
def create_test_universe(K: int, alpha: float, labels: List[str] = None, backend: Literal['numpy', 'jax'] = 'numpy') -> Universe:
    if labels is None:
        labels = [str(i) for i in range(K)]
    return Universe(K=K, alpha=alpha, labels=tuple(labels), backend=backend)
def generate_mock_data(groups: Dict[str, Dict[str, int]], event_labels: List[str], group_col: str = 'group', event_col: str = 'event') -> pd.DataFrame:
    records = []
    for group_name, counts in groups.items():
        for event, count in counts.items():
            if event not in event_labels:
                raise ValueError(f"Event '{event}' in group '{group_name}' is not in event_labels.")
            for _ in range(count):
                records.append({group_col: group_name, event_col: event})
    return pd.DataFrame(records)
