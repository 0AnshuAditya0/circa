import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from pipeline.config import CIRCAConfig
from pipeline.circa_engine import CIRCAEngine

@dataclass
class LatencyReport:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    meets_30fps: bool
    component: str

class LatencyProfiler:

    def __init__(self, engine: CIRCAEngine, config: CIRCAConfig):
        self.engine = engine
        self.config = config

    def _measure(self, fn: Callable, *args: Any, n_runs: int = 100, warmup: int = 10) -> LatencyReport:
        for _ in range(warmup):
            fn(*args)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - start) * 1000.0)

        times_np = np.array(times)
        mean_ms = float(np.mean(times_np))
        
        return LatencyReport(
            mean_ms=mean_ms,
            std_ms=float(np.std(times_np)),
            p50_ms=float(np.percentile(times_np, 50)),
            p95_ms=float(np.percentile(times_np, 95)),
            p99_ms=float(np.percentile(times_np, 99)),
            meets_30fps=mean_ms < 33.3,
            component=fn.__name__
        )

    def profile_fast_loop(self, n_frames: int = 500) -> LatencyReport:
        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_tensor = torch.zeros((1, 3, 256, 256)).to(self.engine.device)
        
        with torch.no_grad():
            cnn_a_out = self.engine.cnn_a(dummy_tensor)
            cnn_b_out = self.engine.cnn_b(dummy_tensor)

        def fast_step():
            self.engine._fast_loop(1, dummy_frame, dummy_tensor, cnn_a_out, cnn_b_out)

        report = self._measure(fast_step, n_runs=n_frames, warmup=10)
        report.component = "Fast Loop (Perception + Causal Query)"
        return report

    def profile_slow_loop(self, n_runs: int = 20) -> LatencyReport:
        dummy_data = np.random.randn(500, self.config.model.latent_dim).astype(np.float32)
        
        def slow_step():
            self.engine.structure_learner.fit(dummy_data)

        report = self._measure(slow_step, n_runs=n_runs, warmup=2)
        report.component = "Slow Loop (NOTEARS DAG Retraining)"
        return report

    def profile_full_pipeline(self, n_frames: int = 500) -> LatencyReport:
        dummy_tensor = torch.zeros((1, 3, 256, 256)).to(self.engine.device)
        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)

        def full_step():
            with torch.no_grad():
                out_a = self.engine.cnn_a(dummy_tensor)
                out_b = self.engine.cnn_b(dummy_tensor)
            self.engine._fast_loop(1, dummy_frame, dummy_tensor, out_a, out_b)

        report = self._measure(full_step, n_runs=n_frames, warmup=10)
        report.component = "Full Pipeline Integration"
        return report

    def save_and_print_results(self, reports: list[LatencyReport], out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_data = [asdict(r) for r in reports]
        with open(out_dir / "latency_results.json", "w") as f:
            json.dump(out_data, f, indent=4)

        console = Console()
        table = Table(title="CIRCA Latency Benchmarks", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Mean (ms)", style="blue")
        table.add_column("P95 (ms)", style="blue")
        table.add_column("Meets 30 FPS", justify="center")

        for r in reports:
            fps_status = "[green]YES[/green]" if r.meets_30fps else "[red]NO[/red]"
            table.add_row(r.component, f"{r.mean_ms:.2f}", f"{r.p95_ms:.2f}", fps_status)

        console.print(table)