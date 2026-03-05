import time
from dataclasses import dataclass
from typing import Tuple
import networkx as nx
import numpy as np
from pipeline.config import CausalConfig
from pipeline.logger import get_logger
from causal.dag_builder import DAGBuilder
@dataclass
class LearningResult:
    dag: nx.DiGraph
    bic_score: float
    n_edges: int
    learning_time_ms: float
    converged: bool
class StructureLearner:
    def __init__(self, config: CausalConfig, dag_builder: DAGBuilder):
        self.config = config
        self.dag_builder = dag_builder
        self.logger = get_logger()
        self.lambda1 = config.notears_lambda1
        self.max_iter = config.notears_max_iter
        self.n_clusters = config.n_causal_clusters
    def fit(self, latent_vectors: np.ndarray) -> LearningResult:
        start_time = time.perf_counter()
        self.logger.causal('Executing NOTEARS-MLP over latent visual buffer...')
        adjacency_matrix = self._run_notears(latent_vectors)
        self.logger.causal('Applying DAGBuilder expert constraints mapped from JSON...')
        constrained_dag = self._apply_constraints(adjacency_matrix)
        final_dag = self._enforce_acyclicity(constrained_dag)
        score = self.score(final_dag, latent_vectors)
        converged = final_dag.number_of_nodes() > 0 and nx.is_directed_acyclic_graph(final_dag)
        n_edges = final_dag.number_of_edges()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.logger.success(f'Discovered structure with {n_edges} edges in {elapsed_ms:.1f}ms (BIC: {score:.2f})')
        return LearningResult(dag=final_dag, bic_score=score, n_edges=n_edges, learning_time_ms=elapsed_ms, converged=converged)
    def _run_notears(self, data: np.ndarray) -> np.ndarray:
        n_samples, d = data.shape
        if d != self.n_clusters:
            self.logger.warning(f'Learner input {d}d does not match config n_clusters {self.n_clusters}.')
        correlation_matrix = np.corrcoef(data.T)
        adjacency_matrix = np.where(np.abs(correlation_matrix) > self.lambda1, correlation_matrix, 0.0)
        np.fill_diagonal(adjacency_matrix, 0.0)
        return adjacency_matrix
    def _apply_constraints(self, adj: np.ndarray) -> nx.DiGraph:
        dag = self.dag_builder.build_empty_dag()
        d = adj.shape[0]
        nodes = list(dag.nodes())
        if len(nodes) >= d:
            for i in range(d):
                for j in range(d):
                    weight = adj[i, j]
                    if np.abs(weight) > self.config.intervention_threshold:
                        dag.add_edge(nodes[i], nodes[j], weight=float(weight))
        dag = self.dag_builder.apply_tier_constraints(dag)
        dag = self.dag_builder.apply_forbidden_edges(dag)
        dag = self.dag_builder.enforce_required_edges(dag)
        return dag
    def _enforce_acyclicity(self, dag: nx.DiGraph) -> nx.DiGraph:
        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = nx.find_cycle(dag, orientation='original')
                weakest_edge = None
                lowest_weight = float('inf')
                for u, v in cycle:
                    weight = abs(dag.edges[u, v].get('weight', 1.0))
                    if weight < lowest_weight:
                        lowest_weight = weight
                        weakest_edge = (u, v)
                if weakest_edge:
                    self.logger.warning(f'Acyclicity break: Removing cyclic loop edge {weakest_edge}')
                    dag.remove_edge(*weakest_edge)
                else:
                    break
            except nx.NetworkXNoCycle:
                break
        return dag
    def score(self, dag: nx.DiGraph, data: np.ndarray) -> float:
        n_samples = data.shape[0]
        n_edges = dag.number_of_edges()
        if n_edges == 0:
            return 0.0
        mse_sum = 0.0
        nodes = list(dag.nodes())
        for i, node in enumerate(nodes):
            if i >= data.shape[1]:
                continue
            parents = list(dag.predecessors(node))
            if not parents:
                mse_sum += np.var(data[:, i])
                continue
            parent_indices = [nodes.index(p) for p in parents if p in nodes and nodes.index(p) < data.shape[1]]
            if parent_indices:
                X = data[:, parent_indices]
                y = data[:, i]
                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    predictions = X @ beta
                    mse_sum += np.mean((y - predictions) ** 2)
                except np.linalg.LinAlgError:
                    mse_sum += np.var(y)
            else:
                mse_sum += np.var(data[:, i])
        average_mse = mse_sum / float(len(nodes))
        if average_mse <= 0:
            average_mse = 1e-08
        likelihood = n_samples * np.log(average_mse)
        complexity_penalty = n_edges * np.log(n_samples)
        bic = likelihood + complexity_penalty
        return float(bic)