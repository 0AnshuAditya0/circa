# CIRCA

CIRCA (Causal Interventional Real-time Computer vision Anomaly detection) is a Structural Causal Model framework for explainable visual anomaly detection.

## Architecture

```text
+-------------------------------------------------+
|               Dashboard (FastAPI + React)       |
+-------------------------------------------------+
                         |
+-------------------------------------------------+
|                Explanation Layer                |
|           (Grad-CAM++ & Causal Heatmaps)        |
+-------------------------------------------------+
                         |
+-------------------------------------------------+
|                  Causal Layer                   |
|       (NOTEARS-MLP, DoWhy, Structural DAG)      |
+-------------------------------------------------+
                         |
+-------------------------------------------------+
|                Perception Layer                 |
|    (C++ OpenCV Ingestion, Dual Encoder CNNs)    |
+-------------------------------------------------+
```

## Quickstart

1. Install requirements:
   `pip install -r requirements.txt`
2. Run backend:
   `uvicorn dashboard.backend:app --reload`

## Abstract
[Placeholder for Abstract]

## Results
[Placeholder for Results]

## Citation
[Placeholder for Citation]
