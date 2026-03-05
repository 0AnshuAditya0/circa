"""
Configuration mapping loaded from environment variables and code-first defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", str(BASE_DIR / "data" / "raw")))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", str(BASE_DIR / "data" / "processed")))
MODEL_CHECKPOINTS = Path(os.getenv("MODEL_CHECKPOINTS", str(BASE_DIR / "perception" / "checkpoints")))

# Model hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 32
LR = 1e-4
LATENT_DIM = 64
BETA = 4.0

# Causal params
DAG_UPDATE_INTERVAL = 500
N_CAUSAL_CLUSTERS = 16
INTERVENTION_THRESHOLD = 0.15

# Stream params
FPS = 30
BUFFER_SIZE = 10

# Device auto-detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
