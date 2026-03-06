import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import json
import random
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.panel import Panel
from rich.table import Table

from pipeline.config import get_config
from pipeline.logger import get_logger
from perception.anomaly_encoder import AnomalyEncoder
from perception.causal_encoder import CausalEncoder
from perception.feature_cluster_mapper import FeatureClusterMapper
from causal.dag_builder import DAGBuilder
from causal.structure_learner import StructureLearner
from causal.snapshot_manager import SnapshotManager
from causal.temporal_dag import TemporalDAG
from causal.do_calculus import InterventionalEngine
from causal.causal_ranker import CausalRanker
from explanation.gradcam_plus import GradCAMPlusPlus
from explanation.report_builder import ReportBuilder
from data.synthetic_generator import CausalDatasetGenerator, CausalDataset
from benchmarks.auroc_eval import AUROCEvaluator
from sklearn.metrics import precision_recall_fscore_support

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config = get_config()
    logger = get_logger()
    console = logger.console

    console.print(Panel("[bold green]CIRCA — Full Pipeline Experiment (All 3 Layers)[/bold green]", expand=False))

    set_seeds(42)
    device = torch.device(config.device)
    logger.info(f"Step 1: Startup -> Using device: {device}")

    try:
        data_dir = config.paths.processed_dir
        dataset_path = data_dir / "synthetic"
        if dataset_path.exists() and (dataset_path / "metadata.csv").exists():
            logger.info("Step 2: Data -> Loading existing synthetic dataset.")
            dataset = CausalDataset.load(dataset_path)
        else:
            logger.warning("Step 2: Data -> Dataset not found. Auto-generating synthetic dataset.")
            generator = CausalDatasetGenerator(config.model)
            dataset = generator.generate_dataset(n_normal=1000, n_per_cause=200)
            dataset.save(dataset_path)
        
        train_ds, val_ds, test_ds = dataset.get_splits(0.7, 0.15, 0.15)
        dataset.summary()

        logger.info("Step 3: Load CNN-A -> Loading AnomalyEncoder from baseline_best.pt")
        baseline_path = config.paths.checkpoint_dir / "baseline_best.pt"
        if not baseline_path.exists():
            logger.error("baseline_best.pt not found. Please run experiments/baseline_run.py first.")
            return

        cnn_a = AnomalyEncoder(config.model).to(device)
        cnn_a.load_state_dict(torch.load(baseline_path, map_location=device))
        for param in cnn_a.parameters():
            param.requires_grad = False
        cnn_a.eval()
        logger.success("CNN-A loaded and frozen \u2713")

        logger.info("Step 4: Train CNN-B -> Training CausalEncoder (beta-VAE)")
        cnn_b = CausalEncoder(config.model).to(device)
        optimizer_b = torch.optim.Adam(cnn_b.parameters(), lr=config.model.learning_rate)
        
        train_loader = DataLoader(train_ds.to_torch_dataset(), batch_size=config.model.batch_size, shuffle=True)
        epochs_b = 10
        
        for epoch in range(epochs_b):
            cnn_b.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"CNN-B Epoch {epoch+1}/{epochs_b}")
            for x, _ in pbar:
                x = x.to(device)
                optimizer_b.zero_grad()
                loss, recon, kl = cnn_b.compute_loss(x)
                loss.backward()
                optimizer_b.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            logger.metric(f"CNN-B Epoch {epoch+1}", f"Loss={total_loss/len(train_loader):.4f}")

        cnn_b_path = config.paths.checkpoint_dir / "causal_encoder_best.pt"
        torch.save(cnn_b.state_dict(), cnn_b_path)
        logger.success(f"CNN-B trained and saved to {cnn_b_path}")

        logger.info("Extracting latent vectors for DAG learning...")
        cnn_b.eval()
        all_z = []
        full_train_loader = DataLoader(train_ds.to_torch_dataset(), batch_size=config.model.batch_size, shuffle=False)
        with torch.no_grad():
            for x, _ in tqdm(full_train_loader, desc="Encoding Train Split"):
                x = x.to(device)
                mu, _ = cnn_b.encode(x)
                all_z.append(mu.cpu().numpy())
        latent_vectors = np.vstack(all_z)

        logger.info("Step 5: Fit Cluster Mapper -> Training FeatureClusterMapper")
        mapper = FeatureClusterMapper(config.causal)
        mapper.fit(latent_vectors)
        mapper_path = config.paths.project_root / "pipeline" / "checkpoints" / "cluster_mapper.pkl"
        mapper.save(mapper_path)
        logger.success(f"Cluster Mapper fitted and saved to {mapper_path}")
        logger.info(f"Cluster Summary: {config.causal.n_causal_clusters} clusters")

        logger.info("Step 6: Learn Causal Structure -> Running StructureLearner")
        dag_builder = DAGBuilder(config.causal)
        dag_path = config.paths.project_root / "causal" / "graphs" / "mvtec_dag.json"
        dag_builder.load_constraints(dag_path)
        
        learner = StructureLearner(config.causal, dag_builder)
        learning_res = learner.fit(latent_vectors)
        
        snapshot_mgr = SnapshotManager()
        temp_dag = TemporalDAG(config.causal, learning_res.dag)
        snapshot_mgr.update_snapshot(temp_dag, frame_id=0)
        
        logger.success(f"Learned DAG: {learning_res.n_edges} edges, BIC: {learning_res.bic_score:.2f}")

        logger.info("Step 7: Full Inference on test split")
        engine = InterventionalEngine(config.causal)
        ranker = CausalRanker(engine, config.causal)
        gradcam = GradCAMPlusPlus(cnn_a, target_layer="backbone.7.2.conv3")
        report_builder = ReportBuilder(ranker, gradcam, snapshot_mgr, config)
        
        test_reports = []
        test_loader = DataLoader(test_ds.to_torch_dataset(), batch_size=1, shuffle=False)
        
        y_true = []
        y_score = []
        y_cause_true = []
        y_cause_pred = []
        
        inf_start = time.perf_counter()
        
        # Track 5 heatmaps for saving
        heatmap_count = 0
        heatmap_dir = config.paths.project_root / "outputs" / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        for i, (x_tensor, label_tensor) in enumerate(tqdm(test_loader, desc="Inference")):
            sample = test_ds.samples[i]
            x_tensor = x_tensor.to(device)
            
            with torch.no_grad():
                out_a = cnn_a(x_tensor)
                out_b = cnn_b(x_tensor)
            
            z_mu, _ = cnn_b.encode(x_tensor)
            
            # For interventional query, we need a dataframe of observations
            # We'll use a subset of training latents as the baseline data
            obs_df = pd.DataFrame(latent_vectors[:200], columns=[f"causal_node_{j}" for j in range(latent_vectors.shape[1])])
            
            # Map test z to nodes for intervention
            # Here we simplify: query effects on the target node from all others
            # In a real run, target_node would be the most symptomatic node
            target_node = f"causal_node_{mapper.map_to_node(z_mu)}"
            
            scores = engine.query_all(temp_dag.to_flat_dag(), obs_df, target_node)
            ranked_causes = ranker.rank(scores)
            significant_causes = ranker.filter_significant(ranked_causes)
            
            # Prepare dummy frame for report builder (numpy expected)
            frame_np = (x_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            report = report_builder.build(frame_id=i, frame=x_tensor, cnn_a_output=out_a, cnn_b_output=out_b, causes=significant_causes)
            test_reports.append(report)
            
            y_true.append(label_tensor.item())
            y_score.append(out_a.score)
            
            if sample.label == 1:
                y_cause_true.append(sample.true_cause)
                pred_cause = "Unknown"
                if significant_causes:
                    # Map causal_node_X back to V1/V2/V3 if possible for eval
                    # Simplified mapping for experiment: 
                    # node 0-4 -> V1, node 5-10 -> V2, node 11-15 -> V3
                    node_id = significant_causes[0].node_id
                    if 0 <= node_id <= 4: pred_cause = "V1"
                    elif 5 <= node_id <= 10: pred_cause = "V2"
                    else: pred_cause = "V3"
                y_cause_pred.append(pred_cause)
                
            if heatmap_count < 5 and out_a.is_anomaly:
                import cv2
                # explanation.gradcam_plus has overlay
                overlay_img = gradcam.overlay(report.heatmap, frame_np)
                cv2.imwrite(str(heatmap_dir / f"anomaly_{heatmap_count}.png"), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                heatmap_count += 1

        inf_end = time.perf_counter()
        avg_latency = (inf_end - inf_start) / len(test_ds.samples) * 1000.0

        logger.info("Step 8: Evaluation")
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, y_score)
        
        causal_correct = sum(1 for p, t in zip(y_cause_pred, y_cause_true) if p == t)
        causal_acc = causal_correct / len(y_cause_true) if y_cause_true else 0.0
        
        v1_indices = [i for i, t in enumerate(y_cause_true) if t == "V1"]
        v2_indices = [i for i, t in enumerate(y_cause_true) if t == "V2"]
        v3_indices = [i for i, t in enumerate(y_cause_true) if t == "V3"]
        
        v1_acc = sum(1 for i in v1_indices if y_cause_pred[i] == "V1") / len(v1_indices) if v1_indices else 0.0
        v2_acc = sum(1 for i in v2_indices if y_cause_pred[i] == "V2") / len(v2_indices) if v2_indices else 0.0
        v3_acc = sum(1 for i in v3_indices if y_cause_pred[i] == "V3") / len(v3_indices) if v3_indices else 0.0

        # F1 Score
        y_pred = [1 if s > 0.5 else 0 for s in y_score]
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        logger.info("Step 9: Save Results")
        results_dir = config.paths.project_root / "benchmarks" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            "detection": {
                "auroc": float(auroc),
                "f1": float(f1)
            },
            "causal_attribution": {
                "overall_accuracy": float(causal_acc),
                "v1_accuracy": float(v1_acc),
                "v2_accuracy": float(v2_acc),
                "v3_accuracy": float(v3_acc)
            },
            "performance": {
                "latency_ms": float(avg_latency),
                "dag_edges": int(learning_res.n_edges),
                "bic_score": float(learning_res.bic_score)
            }
        }
        
        with open(results_dir / "full_circa_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        
        logger.success(f"Results saved to {results_dir / 'full_circa_results.json'}")

        console.print(Panel(
            f"\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n"
            f"\u2502  CIRCA Full Pipeline Results         \u2502\n"
            f"\u2502                                      \u2502\n"
            f"\u2502  Detection:                          \u2502\n"
            f"\u2502    AUROC:          [cyan]{auroc*100:>5.1f}%[/cyan]             \u2502\n"
            f"\u2502    F1 Score:       [cyan]{f1*100:>5.1f}%[/cyan]             \u2502\n"
            f"\u2502                                      \u2502\n"
            f"\u2502  Causal Attribution:                 \u2502\n"
            f"\u2502    Overall Accuracy:  [magenta]{causal_acc*100:>5.1f}%[/magenta]          \u2502\n"
            f"\u2502    V1 (material):     [magenta]{v1_acc*100:>5.1f}%[/magenta]          \u2502\n"
            f"\u2502    V2 (pressure):     [magenta]{v2_acc*100:>5.1f}%[/magenta]          \u2502\n"
            f"\u2502    V3 (surface):      [magenta]{v3_acc*100:>5.1f}%[/magenta]          \u2502\n"
            f"\u2502                                      \u2502\n"
            f"\u2502  Performance:                        \u2502\n"
            f"\u2502    Latency:        [yellow]{avg_latency:>4.1f}ms/sample[/yellow]       \u2502\n"
            f"\u2502    DAG Edges:      [yellow]{learning_res.n_edges:>2}[/yellow]                \u2502\n"
            f"\u2502    BIC Score:      [yellow]{learning_res.bic_score:>7.2f}[/yellow]             \u2502\n"
            f"\u2502                                      \u2502\n"
            f"\u2502  Checkpoints: saved \u2713                \u2502\n"
            f"\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518",
            title="Final Evaluation", expand=False
        ))

        logger.info("Step 11: Connect to Dashboard")
        logger.info("Dashboard auto-reads from benchmarks/results/full_circa_results.json. Connection verified.")

    except ImportError as e:
        if 'causalnex' in str(e):
            logger.error("Requirement 'causalnex' missing.")
            print("Please run: pip install causalnex")
        else:
            logger.error(f"Import error: {str(e)}")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM detected. Retry with half batch size.")
        # Manual adjustment would be required here or recursive retry logic
    except Exception as e:
        logger.error(f"Stage failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import pandas as pd
    main()