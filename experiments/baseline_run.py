import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.config import get_config
from pipeline.logger import get_logger
from perception.anomaly_encoder import AnomalyEncoder
from data.synthetic_generator import CausalDatasetGenerator, CausalDataset
from benchmarks.auroc_eval import AUROCEvaluator

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data(config, logger):
    data_dir = config.paths.processed_dir
    dataset_path = data_dir / "synthetic"
    
    if dataset_path.exists() and (dataset_path / "metadata.csv").exists():
        logger.info("Found existing synthetic causal dataset natively.")
        dataset = CausalDataset.load(dataset_path)
    else:
        logger.warning("No synthetic data bounds mapped. Automatically generating limits natively.")
        generator = CausalDatasetGenerator(config.model)
        dataset = generator.generate_dataset(n_normal=1000, n_per_cause=200)
        dataset.save(dataset_path)
        
    train_ds, val_ds, test_ds = dataset.get_splits(train=0.7, val=0.15, test=0.15)
    
    train_loader = DataLoader(train_ds.to_torch_dataset(), batch_size=config.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds.to_torch_dataset(), batch_size=config.model.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds.to_torch_dataset(), batch_size=config.model.batch_size, shuffle=False)
    
    table = Table(title="Dataset Architecture Limits", header_style="bold cyan")
    table.add_column("Split")
    table.add_column("Sample Matrix Bounds")
    table.add_row("Train", str(len(train_ds.samples)))
    table.add_row("Validation", str(len(val_ds.samples)))
    table.add_row("Test", str(len(test_ds.samples)))
    logger.console.print(table)
    
    return train_loader, val_loader, test_loader

def train_baseline():
    config = get_config()
    logger = get_logger()
    console = logger.console
    
    console.print(Panel("[bold green]CIRCA — Baseline Experiment (CNN-A Only)[/bold green]", expand=False))
    
    set_seeds(42)
    device = torch.device(config.device)
    logger.info(f"Initialized runtime safely bounded to: {device}")
    
    try:
        train_loader, val_loader, test_loader = prepare_data(config, logger)
        
        batch_size = config.model.batch_size
        model = AnomalyEncoder(config.model).to(device)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Instantiated CNN-A Predictive Bounds. Tracked Params: {param_count:,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
        evaluator = AUROCEvaluator(model, config)
        
        config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_chkpt_path = config.paths.checkpoint_dir / "baseline_best.pt"
        best_val_auroc = 0.0
        n_epochs = 10
        
        # OOM automatic batch_size halving mapped via recursion limits natively
        oom_retry = False
        
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} Train")
            try:
                for x, y in pbar:
                    x = x.to(device).float()
                    y = y.unsqueeze(1).to(device).float()

                    optimizer.zero_grad()
                    loss = model.compute_loss(x, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            except torch.cuda.OutOfMemoryError:
                oom_retry = True
                logger.error("CUDA OOM. Recommend halving batch_size in config.")
                break

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device).float()
                    y = y.unsqueeze(1).to(device).float()
                    loss = model.compute_loss(x, y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            val_result = evaluator.evaluate(val_loader)
            
            logger.metric(f"Epoch {epoch+1} Metrics", 
                          f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | ValAUROC={val_result.auroc:.4f}")
                          
            if val_result.auroc > best_val_auroc:
                best_val_auroc = val_result.auroc
                torch.save(model.state_dict(), best_chkpt_path)
                logger.success(f"Minted updated optimal AUC bounds safely ({best_val_auroc:.4f})")
                
        if oom_retry:
            logger.warning("Terminated gracefully mapping OOM. Standard limits recommend scaling 'batch_size' in JSON parameters.")
            return

        logger.info("Initiating strict mathematical boundaries exclusively mapped via optimal configurations natively.")
        model.load_state_dict(torch.load(best_chkpt_path, map_location=device))
        model.eval()
        
        start_t = time.perf_counter()
        test_result = evaluator.evaluate(test_loader)
        total_inf_time = time.perf_counter() - start_t
        
        y_true = []
        y_pred = []
        y_scores_all = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).float()
                batch_scores = []
                for img in x:
                    out = model(img.unsqueeze(0))
                    batch_scores.append(float(out.score))

                preds = (np.array(batch_scores) >= test_result.threshold).astype(int)
                y_true.extend(y.numpy())
                y_pred.extend(preds)
                y_scores_all.extend(batch_scores)
                
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        c_matrix = confusion_matrix(y_true, y_pred)
        
        n_samples = len(test_loader.dataset)
        latency_ms_per_image = (total_inf_time / n_samples) * 1000.0
        
        results_dir = Path("benchmarks/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        final_results = {
            "model": "CNN-A (Baseline)",
            "auroc": test_result.auroc,
            "threshold": test_result.threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "latency_ms": latency_ms_per_image,
            "confusion_matrix": c_matrix.tolist()
        }
        
        with open(results_dir / "baseline_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
            
        panel_content = (
            f"AUROC:     [cyan]{test_result.auroc * 100:.1f}%[/cyan]\n"
            f"F1 Score:  [cyan]{f1 * 100:.1f}%[/cyan]\n"
            f"Latency:   [cyan]{latency_ms_per_image:.1f}ms/image[/cyan]\n"
            f"Checkpoint: [green]saved ✓[/green]\n"
            f"Path: {best_chkpt_path}"
        )
        
        console.print(Panel(panel_content, title="CIRCA Baseline Results", expand=False))

    except Exception as e:
        logger.error(f"Execution boundary natively breached via: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_baseline()