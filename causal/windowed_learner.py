from collections import deque
from typing import Tuple
import numpy as np
import torch
from pipeline.config import CausalConfig
from causal.structure_learner import StructureLearner, LearningResult
from pipeline.logger import get_logger
class WindowedLearner:
    def __init__(self, learner: StructureLearner, config: CausalConfig):
        self.learner = learner
        self.config = config
        self.logger = get_logger()
        self.interval = config.dag_update_interval
        self.buffer_size = self.interval
        self._latent_buffer: deque = deque(maxlen=self.buffer_size)
        self.last_update_frame = 0
    def update(self, frame_id: int, latent_vector: torch.Tensor) -> Tuple[bool, LearningResult]:
        np_vector = latent_vector.detach().cpu().numpy()
        if np_vector.ndim == 1:
            self._latent_buffer.append(np_vector)
        else:
            self._latent_buffer.append(np_vector.flatten())
        if not self._should_update(frame_id):
            return (False, None)
        self.logger.causal(f'Window threshold breached at Frame {frame_id}. Forcing NOTEARS-MLP recomputation.')
        data_window = self._collect_window()
        self.logger.info(f'Batched training over {len(data_window)} recent vectors...')
        result = self.learner.fit(data_window)
        self.last_update_frame = frame_id
        return (True, result)
    def _should_update(self, frame_id: int) -> bool:
        if frame_id == 0:
            return False
        return frame_id % self.interval == 0
    def _collect_window(self) -> np.ndarray:
        return np.array(self._latent_buffer)