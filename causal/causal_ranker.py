from dataclasses import dataclass
from typing import Dict, List
from pipeline.config import CausalConfig
from causal.do_calculus import InterventionalEngine
from pipeline.logger import get_logger
@dataclass
class CauseEntry:
    node_id: int
    node_name: str
    probability: float
    intervention_delta: float
    is_primary: bool
class CausalRanker:
    def __init__(self, engine: InterventionalEngine, config: CausalConfig):
        self.engine = engine
        self.config = config
        self.logger = get_logger()
        self.threshold = config.intervention_threshold
    def rank(self, intervention_scores: Dict[str, float]) -> List[CauseEntry]:
        if not intervention_scores:
            return []
        total_delta = sum(intervention_scores.values())
        if total_delta <= 1e-08:
            return []
        entries = []
        sorted_scores = sorted(intervention_scores.items(), key=lambda item: item[1], reverse=True)
        for idx, (node_name, delta) in enumerate(sorted_scores):
            try:
                base_name = node_name.split('_t')[0]
                node_id = int(base_name.split('_')[-1])
            except (ValueError, IndexError):
                node_id = -1
            prob = float(delta / total_delta)
            is_primary = idx == 0
            entry = CauseEntry(node_id=node_id, node_name=node_name, probability=prob, intervention_delta=delta, is_primary=is_primary)
            entries.append(entry)
        return entries
    def filter_significant(self, causes: List[CauseEntry]) -> List[CauseEntry]:
        significant = [c for c in causes if c.intervention_delta > self.threshold]
        if significant and (not any((c.is_primary for c in significant))):
            significant[0].is_primary = True
        if significant:
            top_cause = significant[0]
            self.logger.causal(f'Attributed primary anomaly cause to {top_cause.node_name} ({top_cause.probability:.1%} confidence).')
        return significant