# CIRCA: Causal Interventional Real-time Computer Vision Anomaly Detection
A structural causal framework mapping visual defects to deterministic root causes using two-speed interventional queries.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Research_Preview-orange.svg)]()
[![Venue](https://img.shields.io/badge/Venue-NeurIPS_2026_CLeaR_(Submitted)-purple.svg)]()

Modern visual anomaly detection models excel at identifying defects but fail to explain why they occur. Standard attribution methods like Grad-CAM highlight anomalous pixels without providing actionable, physical insight into the underlying failure mechanics. CIRCA solves this by unifying representation learning with Pearl's structural causal frameworks. By implementing a strict dual-encoder architecture—separating the anomaly detector from a beta-VAE causal feature extractor—CIRCA constructs a Dynamic Bayesian Network (DBN) mapping visual latent spaces to physical domain tiers. A novel two-speed architecture executes real-time do-calculus interventions against Thread-Safe snapshots of the topology, producing spatial heatmaps and deterministic interventional deltas at 30 FPS. Evaluated on MVTec AD, CIRCA preserves XX% baseline AUROC while delivering definitive interventional rank ordering of root causes. Accompanying this architecture is a novel, mathematically rigorous synthetic causal image dataset explicitly designed for benchmarking structural discovery accuracy.

## Architecture & Dynamics

```text
[INPUT STREAM] ──(30 FPS)──> [ LAYER 1: PERCEPTION ]
                                   │
                                   ├─> CNN-A (Detection) ──────────────┐
                                   │                                   │
                                   └─> CNN-B (Causal beta-VAE) ──┐     │
                                                                 │     │
                             [ LAYER 2: HYBRID SCM ]             │     │
                                                                 │     │
   [ SLOW LOOP: DAG LEARNING ]                 [ FAST LOOP: INFERENCE ]│
   Every N frames:                             Every 1 frame:          │
   1. Collect Latent Buffer                    1. Read RLock DAG       │
   2. Execute NOTEARS-MLP                      2. DoWhy Regression     │
   3. Apply JSON Constraints                   3. Compute Deltas       │
   4. Enforce Acyclicity                       4. Rank Causes          │
   5. Update Snapshot Manager                  5. Return Explanations  │
            ^                                           │              │
            └──────────────(RLock Bridge)───────────────┘              │
                                                                       │
                             [ LAYER 3: EXPLANATION ] <────────────────┘
                                   │
                                   ├─> Grad-CAM++ (Spatial Heatmap)
                                   └─> CausalRanker (Interventional Delta)
                                   │
[DUAL-MODE UI] <───────────────────┘
```

## Contributions

* **Dual Encoder Independence:** Proves that isolating the predictive representation (CNN-A) from the causal representation (CNN-B) mathematically prevents structural bias during interventional reasoning.
* **Two-Speed Hybrid Topology:** Introduces an asynchronous structural discovery loop executing expensive NOTEARS-MLP optimization natively without blocking rapid 30 FPS visual inference limits.
* **Latent-to-DAG Discretization:** Formalizes a direct mapping bridge condensing continuous beta-VAE distributions into discrete, physically bounded Dynamic Bayesian Network nodes using unrolled temporal slices.
* **Synthetic Causal Benchmark Validation:** Releases a procedurally generated structural image dataset strictly parameterized by known mathematical Do-Calculus rules, quantifying absolute topological discovery accuracy.

## Quickstart

```bash
git clone https://github.com/0AnshuAditya0/circa.git
cd circa
pip install -r requirements.txt

# Synthesize ground-truth causal validation dataset
python data/synthetic_generator.py

# Execute the unconstrained standard baseline
python experiments/baseline_run.py

# Execute the full CIRCA architecture
python experiments/full_circa_run.py
```

## Results

Validation bounds computed across the synthetic causal benchmark and the MVTec AD industrial dataset parameters.

| Method | AUROC | Causal Accuracy | Inference Latency |
| :--- | :--- | :--- | :--- |
| Baseline (ResNet50) | XX% | N/A | XXms |
| **CIRCA (Ours)** | **XX%** | **XX%** | **XXms** |

## Project Structure

```text
circa/
├── benchmarks/               # AUROC, Latency, and Ablation tracking
├── causal/                   # SCM Logic, DoWhy bindings, and Two-Speed loops
│   ├── dag_builder.py        # Constraint mapping via JSON heuristics
│   ├── do_calculus.py        # Interventional do() query execution
│   └── structure_learner.py  # NOTEARS-MLP continuous optimization
├── data/                     # Data ingestion and synthetic generation
├── explanation/              # Spatial heatmaps and report generation
├── perception/               # Deep visual architectures
│   ├── anomaly_encoder.py    # CNN-A: Predictive bounds
│   └── causal_encoder.py     # CNN-B: Disentangled representations
└── pipeline/                 # Core real-time asynchronous engine
```

## Citation

```bibtex
@article{aditya2026circa,
  title={CIRCA: Causal Interventional Real-time Computer Vision Anomaly Detection},
  author={Aditya, Anshu},
  journal={CLeaR Workshop, NeurIPS 2026},
  year={2026}
}
```

## Acknowledgements

Built on MVTec AD, PyTorch, DoWhy, and OpenCV.
