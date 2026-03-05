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
    def query(self, dag: nx.DiGraph, obs_data: pd.DataFrame, target_node: str, intervention_node: str) -> float:
        if intervention_node not in dag.nodes or target_node not in dag.nodes:
            return 0.0
        if not nx.has_path(dag, intervention_node, target_node):
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
        for node in dag.nodes():
            if node == target_node:
                continue
            effect = self.query(dag, obs_data, target_node, node)
            scores[node] = effect
        return scores