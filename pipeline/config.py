import os
from pathlib import Path
from typing import Tuple, Union
import torch
from dotenv import load_dotenv
from pydantic import BaseModel
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseSettings
    PYDANTIC_V2 = False
load_dotenv()
def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
class PathConfig(BaseModel):
    project_root: Path = Path(__file__).parent.parent.resolve()
    data_dir: Path = project_root / 'data'
    raw_dir: Path = data_dir / 'raw'
    processed_dir: Path = data_dir / 'processed'
    checkpoint_dir: Path = project_root / 'perception' / 'checkpoints'
    log_dir: Path = project_root / 'logs'
    output_dir: Path = project_root / 'output'
class ModelConfig(BaseModel):
    img_size: int = 256
    batch_size: int = 32
    learning_rate: float = 0.0001
    latent_dim: int = 64
    beta: float = 4.0
    anomaly_threshold: float = 0.5
class CausalConfig(BaseModel):
    dag_update_interval: int = 500
    intervention_threshold: float = 0.15
    structure_threshold: float = 0.3
    max_causes: int = 5
    notears_lambda1: float = 0.001
    notears_max_iter: int = 100
    time_slices: int = 3
    n_causal_clusters: int = 16
class StreamConfig(BaseModel):
    source: Union[int, str] = 0
    fps: int = 30
    buffer_size: int = 10
    resolution: Tuple[int, int] = (256, 256)
class DashboardConfig(BaseModel):
    host: str = 'localhost'
    port: int = 8000
    mode: str = 'researcher'
class CIRCAConfig(BaseSettings):
    paths: PathConfig = PathConfig()
    model: ModelConfig = ModelConfig()
    causal: CausalConfig = CausalConfig()
    stream: StreamConfig = StreamConfig()
    dashboard: DashboardConfig = DashboardConfig()
    device: str = get_optimal_device()
    if PYDANTIC_V2:
        model_config = SettingsConfigDict(env_nested_delimiter='__', env_file='.env', extra='ignore')
    else:
        class Config:
            env_nested_delimiter = '__'
            env_file = '.env'
            extra = 'ignore'
_CONFIG_INSTANCE = None
def get_config() -> CIRCAConfig:
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = CIRCAConfig()
    return _CONFIG_INSTANCE