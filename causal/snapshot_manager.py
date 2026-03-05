from threading import RLock
from causal.temporal_dag import TemporalDAG
from pipeline.logger import get_logger
class SnapshotManager:
    def __init__(self):
        self._current_snapshot = None
        self._lock = RLock()
        self.snapshot_age_frames = 0
        self.logger = get_logger()
    def update_snapshot(self, new_dag: TemporalDAG, frame_id: int) -> None:
        with self._lock:
            if self._current_snapshot is None:
                self.logger.success(f'Primary DAG snapshot minted at Frame {frame_id}.')
            else:
                self.logger.info(f'Snapshot updated natively under RLock boundaries. Minted at {frame_id}.')
            self._current_snapshot = new_dag
            self.snapshot_age_frames = 0
    def get_snapshot(self) -> TemporalDAG:
        with self._lock:
            if self._current_snapshot is None:
                raise RuntimeError('DAG Snapshot Manager initialized before ground-truth NOTEARS graph passed.')
            return self._current_snapshot
    def increment_age(self) -> None:
        with self._lock:
            self.snapshot_age_frames += 1
    def get_age(self) -> int:
        with self._lock:
            return self.snapshot_age_frames