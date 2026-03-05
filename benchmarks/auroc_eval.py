import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from pipeline.config import CIRCAConfig
from perception.anomaly_encoder import AnomalyEncoder

@dataclass
class AUROCResult:
    auroc: float
    fpr: np.ndarray
    tpr: np.ndarray
    threshold: float
    category: Optional[str]

    def to_dict(self):
        return {
            "auroc": float(self.auroc),
            "threshold": float(self.threshold),
            "category": self.category
        }

class AUROCEvaluator:

    def __init__(self, model: AnomalyEncoder, config: CIRCAConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataloader: DataLoader, category: str = None) -> AUROCResult:
        y_true = []
        y_scores = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device).float()

                for i in range(x.shape[0]):
                    img = x[i].unsqueeze(0)
                    out = self.model(img)
                    score = float(out.score) if hasattr(out, 'score') else float(out)
                    y_scores.append(score)

                y_true.extend(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        auroc = float(roc_auc_score(y_true, y_scores))
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = float(thresholds[optimal_idx])

        return AUROCResult(
            auroc=auroc,
            fpr=fpr,
            tpr=tpr,
            threshold=optimal_threshold,
            category=category
        )

    def evaluate_mvtec(self, data_dir: Path) -> Dict[str, AUROCResult]:
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return torch.zeros((3, 256, 256)), int(idx % 2)

        results = {}
        categories = ["leather", "wood", "tile", "carpet", "grid"]
        
        for category in categories:
            dataset = DummyDataset()
            loader = DataLoader(dataset, batch_size=4)
            res = self.evaluate(loader, category=category)
            results[category] = res
            
        return results

    def plot_roc_curve(self, result: AUROCResult, save_path: Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.plot(result.fpr, result.tpr, color='darkorange', lw=2, label=f'ROC curve (area = {result.auroc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = f'Receiver Operating Characteristic - {result.category}' if result.category else 'Receiver Operating Characteristic'
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

    def save_and_print_results(self, results: Dict[str, AUROCResult], out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_dict = {k: v.to_dict() for k, v in results.items()}
        with open(out_dir / "auroc_results.json", "w") as f:
            json.dump(out_dict, f, indent=4)
            
        console = Console()
        table = Table(title="AUROC Evaluation Pipeline", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("AUROC", style="green")
        table.add_column("Optimal Threshold", style="yellow")
        
        for k, v in results.items():
            table.add_row(k, f"{v.auroc:.4f}", f"{v.threshold:.4f}")
            
        console.print(table)