from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

class ModelConfig:
    pass

@dataclass
class SyntheticSample:
    image: np.ndarray
    label: int
    true_cause: Optional[str]
    true_cause_tier: Optional[int]
    causal_values: Dict[str, float]
    intervention_strength: float

class CausalDatasetGenerator:

    def __init__(self, config: ModelConfig = None, seed: int = 42):
        self.config = config
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _sample_scm(self, interventions: Dict[str, float] = None) -> Dict[str, float]:
        interventions = interventions or {}
        
        if "V1" in interventions:
            v1 = interventions["V1"]
        else:
            v1 = float(np.random.normal(0.5, 0.1))
            
        if "V2" in interventions:
            v2 = interventions["V2"]
        else:
            v2 = 0.6 * v1 + float(np.random.normal(0, 0.05))
            
        if "V3" in interventions:
            v3 = interventions["V3"]
        else:
            v3 = 0.4 * v1 + 0.5 * v2 + float(np.random.normal(0, 0.05))
            
        return {"V1": v1, "V2": v2, "V3": v3}

    def _apply_v1_effect(self, img: np.ndarray, strength: float) -> np.ndarray:
        mean_val = np.mean(img)
        return (img - mean_val) * (1.0 - 0.6 * strength) + mean_val

    def _apply_v2_effect(self, img: np.ndarray, strength: float) -> np.ndarray:
        streak = np.zeros_like(img)
        for i in range(img.shape[0]):
            streak[i, :, :] = np.sin(i / 10.0) * 0.3 * strength
        return img + streak

    def _apply_v3_effect(self, img: np.ndarray, strength: float) -> np.ndarray:
        y, x = np.ogrid[0:img.shape[0], 0:img.shape[1]]
        cx, cy = np.random.randint(64, 192, size=2)
        radius = 20 + 20 * strength
        mask = (x - cx)**2 + (y - cy)**2 < radius**2
        img[mask] = img[mask] * max(0.0, 1.0 - strength)
        return img

    def _render_image(self, causal_values: Dict[str, float]) -> np.ndarray:
        size = 256
        noise = torch.rand(1, 3, size // 16, size // 16)
        img_tensor = F.interpolate(noise, size=(size, size), mode='bilinear', align_corners=False)
        img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        
        v1, v2, v3 = causal_values["V1"], causal_values["V2"], causal_values["V3"]
        
        if v1 < 0.3:
            img = self._apply_v1_effect(img, min(1.0, (0.3 - v1) / 0.2))
        
        if v2 > 0.6:
            img = self._apply_v2_effect(img, min(1.0, (v2 - 0.6) / 0.3))
            
        if v3 < 0.25:
            img = self._apply_v3_effect(img, min(1.0, (0.25 - v3) / 0.2))
            
        return (img * 255).clip(0, 255).astype(np.uint8)

    def generate_normal(self, n: int) -> List[SyntheticSample]:
        samples = []
        for _ in range(n):
            causal_values = self._sample_scm()
            img = self._render_image(causal_values)
            samples.append(SyntheticSample(
                image=img,
                label=0,
                true_cause=None,
                true_cause_tier=None,
                causal_values=causal_values,
                intervention_strength=0.0
            ))
        return samples

    def generate_anomaly(self, n: int, cause: str) -> List[SyntheticSample]:
        if cause not in ["V1", "V2", "V3"]:
            raise ValueError(f"Invalid cause {cause}. Must be V1, V2, or V3")
            
        samples = []
        tier_map = {"V1": 0, "V2": 1, "V3": 2}
        
        for _ in range(n):
            interventions = {}
            if cause == "V1":
                interventions["V1"] = float(np.random.normal(0.1, 0.05))
            elif cause == "V2":
                interventions["V2"] = float(np.random.normal(0.9, 0.05))
            elif cause == "V3":
                interventions["V3"] = float(np.random.normal(0.05, 0.02))
                
            causal_values = self._sample_scm(interventions)
            img = self._render_image(causal_values)
            
            if cause == "V1":
                strength = 0.5 - causal_values["V1"]
            elif cause == "V2":
                strength = causal_values["V2"] - 0.3
            else:
                strength = 0.35 - causal_values["V3"]
                
            samples.append(SyntheticSample(
                image=img,
                label=1,
                true_cause=cause,
                true_cause_tier=tier_map[cause],
                causal_values=causal_values,
                intervention_strength=float(strength)
            ))
            
        return samples

    def generate_dataset(self, n_normal: int, n_per_cause: int) -> 'CausalDataset':
        samples = []
        samples.extend(self.generate_normal(n_normal))
        samples.extend(self.generate_anomaly(n_per_cause, "V1"))
        samples.extend(self.generate_anomaly(n_per_cause, "V2"))
        samples.extend(self.generate_anomaly(n_per_cause, "V3"))
        
        np.random.shuffle(samples)
        return CausalDataset(samples)


class CausalDataset:
    def __init__(self, samples: List[SyntheticSample]):
        self.samples = samples

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        img_dir = path / "images"
        img_dir.mkdir(exist_ok=True)
        
        metadata = []
        import torchvision
        
        for idx, sample in enumerate(self.samples):
            img_name = f"sample_{idx:05d}.png"
            img_path = img_dir / img_name
            
            tensor_img = torch.from_numpy(sample.image).permute(2, 0, 1).contiguous()
            torchvision.io.write_png(tensor_img, str(img_path))
            
            meta_record = {
                "filename": img_name,
                "label": sample.label,
                "true_cause": sample.true_cause if sample.true_cause else "",
                "true_cause_tier": sample.true_cause_tier if sample.true_cause_tier is not None else "",
                "intervention_strength": sample.intervention_strength,
                "v1_val": sample.causal_values["V1"],
                "v2_val": sample.causal_values["V2"],
                "v3_val": sample.causal_values["V3"]
            }
            metadata.append(meta_record)
            
        df = pd.DataFrame(metadata)
        df.to_csv(path / "metadata.csv", index=False)

    @classmethod
    def load(cls, path: Path) -> 'CausalDataset':
        import torchvision
        path = Path(path)
        df = pd.read_csv(path / "metadata.csv")
        df = df.replace({np.nan: None})
        
        samples = []
        for _, row in df.iterrows():
            img_path = path / "images" / row["filename"]
            tensor_img = torchvision.io.read_image(str(img_path))
            img = tensor_img.permute(1, 2, 0).numpy()
            
            tc = row["true_cause"]
            tct = row["true_cause_tier"]
            if tc == "": tc = None
            if tct == "": tct = None
            
            cv = {
                "V1": float(row["v1_val"]),
                "V2": float(row["v2_val"]),
                "V3": float(row["v3_val"])
            }
            
            samples.append(SyntheticSample(
                image=img,
                label=int(row["label"]),
                true_cause=str(tc) if tc is not None else None,
                true_cause_tier=int(float(tct)) if tct is not None else None,
                causal_values=cv,
                intervention_strength=float(row["intervention_strength"])
            ))
        return cls(samples)

    def get_splits(self, train: float = 0.7, val: float = 0.15, test: float = 0.15) -> Tuple['CausalDataset', 'CausalDataset', 'CausalDataset']:
        if not np.isclose(train + val + test, 1.0):
            raise ValueError("Splits must sum exactly to 1.0")
            
        n = len(self.samples)
        n_train = int(n * train)
        n_val = int(n * val)
        
        return (
            CausalDataset(self.samples[:n_train]),
            CausalDataset(self.samples[n_train:n_train + n_val]),
            CausalDataset(self.samples[n_train + n_val:])
        )

    def summary(self) -> None:
        console = Console()
        table = Table(title="Synthesized Causal Dataset", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value / Count", style="green")
        
        table.add_row("Total Samples", str(len(self.samples)))
        table.add_row("Normal Samples (L=0)", str(sum(1 for s in self.samples if s.label == 0)))
        table.add_row("Anomaly Samples (L=1)", str(sum(1 for s in self.samples if s.label == 1)))
        
        for cause in ["V1", "V2", "V3"]:
            c_count = sum(1 for s in self.samples if s.true_cause == cause)
            table.add_row(f"Anomalies purely from do({cause}=x)", str(c_count))
            
        console.print(table)

    def to_torch_dataset(self) -> torch.utils.data.Dataset:
        class TorchCausalWrapper(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                s = self.samples[idx]
                return torch.from_numpy(s.image).float().permute(2, 0, 1) / 255.0, torch.tensor(s.label)
        return TorchCausalWrapper(self.samples)

@dataclass
class CausalAccuracyReport:
    overall_accuracy: float
    per_cause_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    tier_accuracy: Dict[int, float]

    def summary(self) -> None:
        console = Console()
        table = Table(title="Do-Calculus Explanatory Attribution Accuracy", header_style="bold green")
        table.add_column("Causal Metric", style="cyan")
        table.add_column("Accuracy (%)", style="magenta")
        
        table.add_row("Overall Structural Accuracy", f"{self.overall_accuracy * 100:.1f}%")
        table.add_row("", "")
        for cause, acc in self.per_cause_accuracy.items():
            table.add_row(f"Resolution Accuracy ({cause})", f"{acc * 100:.1f}%")
        table.add_row("", "")
        for tier, acc in self.tier_accuracy.items():
            table.add_row(f"Tier {tier} Hierarchical Match", f"{acc * 100:.1f}%")
            
        console.print(table)

def evaluate_causal_accuracy(predictions: List[str], ground_truth: List[str]) -> CausalAccuracyReport:
    if len(predictions) != len(ground_truth):
        raise ValueError("Mismatched prediction and ground truth bounds.")
        
    pairs = [(p, g) for p, g in zip(predictions, ground_truth) if g is not None]
    
    if not pairs:
        return CausalAccuracyReport(0.0, {}, np.zeros((3, 3)), {})

    correct = sum(1 for p, g in pairs if p == g)
    overall = correct / len(pairs)
    
    causes = ["V1", "V2", "V3"]
    per_cause = {}
    for c in causes:
        c_samples = [p for p, g in pairs if g == c]
        if c_samples:
            per_cause[c] = float(sum(1 for p, g in c_samples if p == g) / len(c_samples))
        else:
            per_cause[c] = 0.0
            
    cm = np.zeros((3, 3), dtype=int)
    for p, g in pairs:
        if p in causes and g in causes:
            cm[causes.index(g), causes.index(p)] += 1
            
    tier_map = {"V1": 0, "V2": 1, "V3": 2}
    tier_acc = {}
    for tier in [0, 1, 2]:
        t_samples = [p for p, g in pairs if tier_map.get(g) == tier]
        if t_samples:
            tier_acc[tier] = float(sum(1 for p, g in t_samples if tier_map.get(p) == tier_map.get(g)) / len(t_samples))
            
    return CausalAccuracyReport(overall_accuracy=overall, per_cause_accuracy=per_cause, confusion_matrix=cm, tier_accuracy=tier_acc)

if __name__ == "__main__":
    console = Console()
    console.print("[bold cyan]Executing CIRCA Synthetic Causal Data Generator...[/bold cyan]")
    generator = CausalDatasetGenerator()
    dataset = generator.generate_dataset(n_normal=1000, n_per_cause=200)
    dataset.summary()
