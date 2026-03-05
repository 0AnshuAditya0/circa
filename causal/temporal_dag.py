from typing import Dict, List, Set, Tuple
import networkx as nx
from pipeline.config import CausalConfig
class TemporalDAG:
    def __init__(self, config: CausalConfig, base_dag: nx.DiGraph):
        self.config = config
        self.base_dag = base_dag
        self.time_slices = config.time_slices
        self.flat_dag = nx.DiGraph()
        self.slices: Dict[int, List[str]] = {}
        self._initialize_dbn()
    def _initialize_dbn(self) -> None:
        for t in range(self.time_slices):
            self.add_time_slice(t)
        self.connect_slices()
    def add_time_slice(self, t: int) -> None:
        slice_nodes = []
        for node in self.base_dag.nodes():
            temporal_node = f'{node}_t{t}'
            slice_nodes.append(temporal_node)
            attrs = self.base_dag.nodes[node]
            self.flat_dag.add_node(temporal_node, **attrs, time=t, base_name=node)
        for u, v in self.base_dag.edges():
            temporal_u = f'{u}_t{t}'
            temporal_v = f'{v}_t{t}'
            weight = self.base_dag.edges[u, v].get('weight', 1.0)
            self.flat_dag.add_edge(temporal_u, temporal_v, weight=weight, type='spatial')
        self.slices[t] = slice_nodes
    def connect_slices(self) -> None:
        if self.time_slices <= 1:
            return
        for t in range(1, self.time_slices):
            prev_nodes = self.slices[t - 1]
            curr_nodes = self.slices[t]
            for prev_n, curr_n in zip(prev_nodes, curr_nodes):
                self.flat_dag.add_edge(prev_n, curr_n, weight=1.0, type='temporal')
    def get_slice(self, t: int) -> nx.DiGraph:
        if t not in self.slices:
            raise IndexError(f'Temporal slice {t} out of bounds.')
        return self.flat_dag.subgraph(self.slices[t]).copy()
    def to_flat_dag(self) -> nx.DiGraph:
        return self.flat_dag.copy()