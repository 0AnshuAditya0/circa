import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from pipeline.config import CausalConfig
class DAGBuilder:
    def __init__(self, config: CausalConfig):
        self.config = config
        self.time_slices = config.time_slices
        self.max_causes = config.max_causes
        self.tiers: List[List[str]] = []
        self.forbidden_edges: Set[Tuple[str, str]] = set()
        self.required_edges: Set[Tuple[str, str]] = set()
        self.node_descriptions: Dict[str, str] = {}
        self.domain = 'unknown'
    def load_constraints(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f'Expert constraints JSON not found at {path}')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.domain = data.get('domain', 'unknown')
        self.tiers = data.get('tiers', [])
        self.node_descriptions = data.get('node_descriptions', {})
        self.forbidden_edges = {(edge[0], edge[1]) for edge in data.get('forbidden_edges', [])}
        self.required_edges = {(edge[0], edge[1]) for edge in data.get('required_edges', [])}

    def get_node_tier(self, node_name: str) -> int | None:
        base_name = node_name.split('_t')[0]
        for tier_idx, nodes in enumerate(self.tiers):
            if base_name in nodes:
                return tier_idx
        return None
    def build_empty_dag(self) -> nx.DiGraph:
        dag = nx.DiGraph()
        for tier_idx, nodes in enumerate(self.tiers):
            for node in nodes:
                desc = self.node_descriptions.get(node, 'Unknown Vector')
                dag.add_node(node, tier=tier_idx, description=desc)
        return dag
    def apply_tier_constraints(self, dag: nx.DiGraph) -> nx.DiGraph:
        edges_to_remove = []
        for u, v in dag.edges():
            tier_u = dag.nodes[u].get('tier', 0)
            tier_v = dag.nodes[v].get('tier', 0)
            time_u = dag.nodes[u].get('time', 0)
            time_v = dag.nodes[v].get('time', 0)
            if time_u == time_v and tier_u > tier_v:
                edges_to_remove.append((u, v))
            if time_u > time_v:
                edges_to_remove.append((u, v))
        dag.remove_edges_from(edges_to_remove)
        return dag
    def apply_forbidden_edges(self, dag: nx.DiGraph) -> nx.DiGraph:
        edges_to_remove = []
        for u, v in dag.edges():
            base_u = dag.nodes[u].get('base_name', u)
            base_v = dag.nodes[v].get('base_name', v)
            if (base_u, base_v) in self.forbidden_edges:
                edges_to_remove.append((u, v))
        dag.remove_edges_from(edges_to_remove)
        return dag
    def enforce_required_edges(self, dag: nx.DiGraph) -> nx.DiGraph:
        for u, v in self.required_edges:
            if dag.has_node(u) and dag.has_node(v):
                dag.add_edge(u, v)
        return dag
    def validate_dag(self, dag: nx.DiGraph) -> bool:
        if dag is None or dag.number_of_nodes() == 0:
            return False
        return nx.is_directed_acyclic_graph(dag)
    def visualize_dag(self, dag: nx.DiGraph) -> None:
        if dag.number_of_nodes() == 0:
            print('DAG is empty. Cannot visualize.')
            return
        plt.figure(figsize=(14, 10))
        pos = nx.multipartite_layout(dag, subset_key='tier')
        color_map = {0: 'lightgreen', 1: 'lightblue', 2: 'lightcoral', 'default': 'lightgrey'}
        node_colors = [color_map.get(dag.nodes[node].get('tier', 'default'), 'lightgrey') for node in dag.nodes()]
        nx.draw_networkx_nodes(dag, pos, node_size=2000, node_color=node_colors, edgecolors='black')
        nx.draw_networkx_edges(dag, pos, arrowsize=20, arrowstyle='->', width=1.5, min_source_margin=15, min_target_margin=15)
        labels = {n: n.replace('_', '\n') for n in dag.nodes()}
        nx.draw_networkx_labels(dag, pos, font_size=9, font_weight='bold', labels=labels)
        plt.title('CIRCA Structural Causal Model Architecture', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    def to_dowhy_model(self, dag: nx.DiGraph) -> str:
        graph_lines = ['digraph {']
        for u, v in dag.edges():
            graph_lines.append(f'  "{u}" -> "{v}";')
        graph_lines.append('}')
        return '\n'.join(graph_lines)