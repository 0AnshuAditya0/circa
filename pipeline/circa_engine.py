import threading
import time
from typing import Union
import cv2
import numpy as np
import pandas as pd
import torch
from pipeline.config import CIRCAConfig, get_config
from pipeline.logger import get_logger, CIRCALogger
from perception.anomaly_encoder import AnomalyEncoder
from perception.causal_encoder import CausalEncoder
from perception.feature_cluster_mapper import FeatureClusterMapper
from causal.dag_builder import DAGBuilder
from causal.temporal_dag import TemporalDAG
from causal.structure_learner import StructureLearner
from causal.windowed_learner import WindowedLearner
from causal.snapshot_manager import SnapshotManager
from causal.do_calculus import InterventionalEngine
from causal.causal_ranker import CausalRanker
from explanation.gradcam_plus import GradCAMPlusPlus
from explanation.report_builder import ReportBuilder, CIRCAReport
class CIRCAEngine:
    def __init__(self, config: CIRCAConfig):
        self.config = config
        self.logger = get_logger()
        self.is_running = False
        self._slow_loop_lock = threading.Lock()
        self.logger.info('Initializing CIRCA Execution Engine...')
        self._build_components()
    def _build_components(self) -> None:
        self.logger.info('Loading Dual Encoders (CNN-A and CNN-B)...')
        self.device = torch.device(self.config.device)
        self.cnn_a = AnomalyEncoder(self.config.model).to(self.device).eval()
        self.cnn_b = CausalEncoder(self.config.model).to(self.device).eval()
        self.logger.info('Configuring Structural Causal Graph limits...')
        self.dag_builder = DAGBuilder(self.config.causal)
        try:
            constraints_path = self.config.paths.project_root / 'causal' / 'graphs' / 'mvtec_dag.json'
            self.dag_builder.load_constraints(constraints_path)
        except Exception as e:
            self.logger.warning(f'Failed loading external DAG constraints schema: {e}. Executing unconstrained.')
        self.cluster_mapper = FeatureClusterMapper(self.config.model)
        self.structure_learner = StructureLearner(self.config.causal, self.dag_builder)
        self.windowed_learner = WindowedLearner(self.structure_learner, self.config.causal)
        self.snapshot_manager = SnapshotManager()
        empty_dag = self.dag_builder.build_empty_dag()
        empty_temporal_dag = TemporalDAG(self.config.causal, empty_dag)
        self.snapshot_manager.update_snapshot(empty_temporal_dag, frame_id=0)
        self.do_engine = InterventionalEngine(self.config.causal)
        self.ranker = CausalRanker(self.do_engine, self.config.causal)
        self.gradcam = GradCAMPlusPlus(self.cnn_a, target_layer='backbone.7.2.conv3')
        self.report_builder = ReportBuilder(self.ranker, self.gradcam, self.snapshot_manager, self.config)
        self.logger.success('All CIRCA structural dependencies configured successfully.')
    def run_stream(self, source: Union[int, str]=None) -> None:
        stream_src = source if source is not None else self.config.stream.source
        cap = cv2.VideoCapture(stream_src)
        if not cap.isOpened():
            self.logger.error(f"FATAL: Unable to resolve video capture source '{stream_src}'")
            return
        self.is_running = True
        frame_id = 0
        self.logger.success(f'STREAM COMMENCED: Engine native bounds initialized securely on {self.device}.')
        try:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning('Upstream visual frame ingestion severed seamlessly.')
                    break
                frame_id += 1
                self.logger.stream('Ingesting perceptual byte array successfully.', frame_id=frame_id)
                target_res = self.config.stream.resolution
                resized = cv2.resize(frame, target_res)
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(rgb_frame).float().permute(2, 0, 1).unsqueeze(0)
                frame_tensor = (frame_tensor / 255.0).to(self.device)
                with torch.no_grad():
                    cnn_a_out = self.cnn_a(frame_tensor)
                    cnn_b_out = self.cnn_b(frame_tensor)
                self._slow_loop(frame_id, cnn_b_out.mu)
                report = self._fast_loop(frame_id, rgb_frame, frame_tensor, cnn_a_out, cnn_b_out)
                if report.is_anomaly:
                    operator_dict = self.report_builder.to_operator_dict(report)
                    self.logger.log_anomaly_report(operator_dict)
                time.sleep(1.0 / self.config.stream.fps)
        except KeyboardInterrupt:
            self.logger.warning('KeyboardInterrupt detected. CIRCA orchestrator aborting sequences...')
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()
    def _fast_loop(self, frame_id: int, original_frame: np.ndarray, tensor_frame: torch.Tensor, cnn_a_out, cnn_b_out) -> CIRCAReport:
        self.snapshot_manager.increment_age()
        active_snapshot: TemporalDAG = self.snapshot_manager.get_snapshot()
        flat_dag = active_snapshot.to_flat_dag()
        causes_list = []
        if cnn_a_out.is_anomaly and flat_dag.number_of_edges() > 0:
            z_np = cnn_b_out.mu.detach().cpu().numpy().flatten()
            node_names = list(flat_dag.nodes())
            dummy_data = {node: np.random.normal(loc=z_np[i % len(z_np)], size=100) for i, node in enumerate(node_names)}
            target_node = 'anomaly_score_output'
            flat_dag.add_node(target_node, tier=99)
            for node in node_names:
                flat_dag.add_edge(node, target_node, weight=1.0)
            dummy_data[target_node] = np.random.normal(loc=cnn_a_out.score, size=100)
            obs_df = pd.DataFrame(dummy_data)
            raw_scores = self.do_engine.query_all(flat_dag, obs_df, target_node)
            causes_list = self.ranker.rank(raw_scores)
            causes_list = self.ranker.filter_significant(causes_list)
        report = self.report_builder.build(frame_id, tensor_frame, cnn_a_out, cnn_b_out, causes_list)
        return report
    def _slow_loop(self, frame_id: int, latent_vector: torch.Tensor) -> None:
        vector_np = latent_vector.detach().cpu().numpy().flatten()
        self.windowed_learner._latent_buffer.append(vector_np)
        if not self.windowed_learner._should_update(frame_id):
            return
        if not self._slow_loop_lock.acquire(blocking=False):
            self.logger.warning(f'Skipping SLOW LOOP trigger at Frame {frame_id} natively - Previous NOTEARS thread logic mathematically still computing limits.')
            return
        window_data = self.windowed_learner._collect_window().copy()
        def background_task():
            try:
                self.logger.causal(f'[SLOW LOOP] Triggered structural recomputations organically at bounds {frame_id} natively...')
                if not self.cluster_mapper.is_fitted:
                    self.cluster_mapper.fit(window_data)
                result = self.structure_learner.fit(window_data)
                if result.converged:
                    new_temporal = TemporalDAG(self.config.causal, result.dag)
                    self.snapshot_manager.update_snapshot(new_temporal, frame_id)
                else:
                    self.logger.warning(f'[SLOW LOOP] NOTEARS matrix math inherently failed to organically converge native bounds limit topology safely.')
            except Exception as e:
                self.logger.error(f'[SLOW LOOP] Fatal continuous optimization metric abort organically limits: {e}')
            finally:
                self._slow_loop_lock.release()
        thread = threading.Thread(target=background_task, name=f'SlowLoop_Thread_F{frame_id}')
        thread.daemon = True
        thread.start()
    def stop(self) -> None:
        if self.is_running:
            self.logger.info('Deactivating CIRCA Engine Bounds securely.')
            self.is_running = False