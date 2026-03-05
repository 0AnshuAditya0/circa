import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import numpy as np
from causal.causal_ranker import CausalRanker, CauseEntry
from causal.snapshot_manager import SnapshotManager
from explanation.gradcam_plus import GradCAMPlusPlus
from pipeline.config import CIRCAConfig
@dataclass
class CIRCAReport:
    frame_id: int
    timestamp: float
    anomaly_score: float
    is_anomaly: bool
    top_causes: List[CauseEntry]
    heatmap: np.ndarray
    dag_snapshot_age: int
    operator_action: str
class ReportBuilder:
    def __init__(self, ranker: CausalRanker, gradcam: GradCAMPlusPlus, snapshot_manager: SnapshotManager, config: CIRCAConfig):
        self.ranker = ranker
        self.gradcam = gradcam
        self.snapshot_manager = snapshot_manager
        self.config = config
    def build(self, frame_id: int, frame: np.ndarray, cnn_a_output: Any, cnn_b_output: Any, causes: List[CauseEntry]=None) -> CIRCAReport:
        heatmap = self.gradcam.generate(frame, target_class=0)
        if causes is None:
            causes = []
        action_directive = 'MONITOR'
        if cnn_a_output.is_anomaly and causes:
            primary_name = causes[0].node_name
            action_directive = f'INSPECT BOUND: {primary_name.upper()}'
        elif cnn_a_output.is_anomaly:
            action_directive = 'MANUAL INVESTIGATION REQUIRED'
        return CIRCAReport(frame_id=frame_id, timestamp=time.time(), anomaly_score=cnn_a_output.score, is_anomaly=cnn_a_output.is_anomaly, top_causes=causes, heatmap=heatmap, dag_snapshot_age=self.snapshot_manager.get_age(), operator_action=action_directive)
    def to_operator_dict(self, report: CIRCAReport) -> Dict[str, Any]:
        primary_cause = report.top_causes[0].node_name if report.top_causes else 'Unknown'
        return {'frame_id': report.frame_id, 'status': 'CRITICAL' if report.is_anomaly else 'NORMAL', 'anomaly_score': round(report.anomaly_score, 4), 'primary_root_cause': primary_cause, 'operator_action': report.operator_action}
    def to_researcher_dict(self, report: CIRCAReport) -> Dict[str, Any]:
        causes_dict = [asdict(cause) for cause in report.top_causes]
        return {'frame_id': report.frame_id, 'timestamp': report.timestamp, 'anomaly_score': report.anomaly_score, 'is_anomaly': report.is_anomaly, 'dag_snapshot_age': report.dag_snapshot_age, 'operator_action': report.operator_action, 'top_causes': causes_dict, 'has_heatmap': report.heatmap is not None}
    def to_json(self, report: CIRCAReport) -> str:
        safe_dict = self.to_researcher_dict(report)
        return json.dumps(safe_dict, indent=2)