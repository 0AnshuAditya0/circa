import time
from typing import Dict
import networkx as nx
import pandas as pd
from dowhy import CausalModel
from pipeline.config import CausalConfig
from causal.dag_builder import DAGBuilder
from pipeline.logger import get_logger
class InterventionalEngine:
    def __init__(self, config: CausalConfig):
        self.config = config
        self.logger = get_logger()
        self._dag_parser = DAGBuilder(config)
    def _filter_dag_to_observed(self, dag: nx.DiGraph, obs_data: pd.DataFrame) -> nx.DiGraph:
        observed_nodes = set(obs_data.columns)
        nodes_to_keep = [n for n in dag.nodes() if n in observed_nodes]
        if not nodes_to_keep:
            return dag.copy()
        return dag.subgraph(nodes_to_keep).copy()
    def query(self, dag: nx.DiGraph, obs_data: pd.DataFrame, target_node: str, intervention_node: str) -> float:
        if intervention_node not in dag.nodes or target_node not in dag.nodes:
            return 0.0
        if not nx.has_path(dag, intervention_node, target_node):
            return 0.0
        dag = self._filter_dag_to_observed(dag, obs_data)
        if len(dag.nodes) < 2:
            return 0.0
        try:
            dowhy_graph_str = self._dag_parser.to_dowhy_model(dag)
            model = CausalModel(data=obs_data, treatment=intervention_node, outcome=target_node, graph=dowhy_graph_str, logging_level='CRITICAL')
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression')
            return float(abs(estimate.value))
        except Exception as e:
            self.logger.warning(f'DoWhy intervention failed on {intervention_node} -> {target_node}: {str(e)}')
            return 0.0
    def query_all(self, dag: nx.DiGraph, obs_data: pd.DataFrame, target_node: str) -> Dict[str, float]:
        scores = {}
        filtered_dag = self._filter_dag_to_observed(dag, obs_data)
        
        nodes = list(filtered_dag.nodes())
        nodes_with_tiers = []
        for n in nodes:
            if n == target_node: continue
            tier = filtered_dag.nodes[n].get('tier', 99)
            nodes_with_tiers.append((n, tier))
        
        nodes_with_tiers.sort(key=lambda x: x[1])
        candidate_nodes = [node for node, tier in nodes_with_tiers][:10]
        
        start_q = time.perf_counter()
        for node in candidate_nodes:
            effect = self.query(filtered_dag, obs_data, target_node, node)
            scores[node] = effect
            
        elapsed = (time.perf_counter() - start_q) * 1000.0
        self.logger.info(f"DoWhy batch query complete: {len(scores)} nodes in {elapsed:.1f}ms")
        return scores