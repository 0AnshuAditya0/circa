import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .config import get_config
LEVEL_SUCCESS = 25
LEVEL_METRIC = 26
LEVEL_CAUSAL = 27
LEVEL_STREAM = 28
logging.addLevelName(LEVEL_SUCCESS, 'SUCCESS')
logging.addLevelName(LEVEL_METRIC, 'METRIC')
logging.addLevelName(LEVEL_CAUSAL, 'CAUSAL')
logging.addLevelName(LEVEL_STREAM, 'STREAM')
class CIRCALogger:
    def __init__(self, name: str='CIRCA'):
        config = get_config()
        self.console = Console()
        self.log_dir = config.paths.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = self.log_dir / f'circa_run_{timestamp}.log'
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        if self._logger.handlers:
            self._logger.handlers.clear()
        rich_handler = RichHandler(console=self.console, show_time=True, show_path=False, rich_tracebacks=True, markup=True)
        rich_handler.setLevel(logging.DEBUG)
        class ConsoleStreamFilter(logging.Filter):
            def __init__(self, parent_logger):
                super().__init__()
                self.parent = parent_logger
            def filter(self, record):
                if record.levelno == LEVEL_STREAM:
                    return self.parent._stream_counter % 100 == 0
                return True
        rich_handler.addFilter(ConsoleStreamFilter(self))
        self._logger.addHandler(rich_handler)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(fmt='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        self._logger.addHandler(file_handler)
        self._stream_counter = 0
    def info(self, msg: str):
        self._logger.info(f'[white]{msg}[/white]', extra={'markup': True})
    def success(self, msg: str):
        self._logger.log(LEVEL_SUCCESS, f'[bold green]✓ {msg}[/bold green]', extra={'markup': True})
    def warning(self, msg: str):
        self._logger.warning(f'[bold yellow]⚠ {msg}[/bold yellow]', extra={'markup': True})
    def error(self, msg: str, exc_info=False):
        self._logger.error(f'[bold red]❌ {msg}[/bold red]', exc_info=exc_info, extra={'markup': True})
    def metric(self, key: str, value: any):
        msg = f'[cyan]{key}[/cyan]: [bold cyan]{value}[/bold cyan]'
        self._logger.log(LEVEL_METRIC, msg, extra={'markup': True})
    def causal(self, msg: str):
        self._logger.log(LEVEL_CAUSAL, f'[bold magenta]∞ {msg}[/bold magenta]', extra={'markup': True})
    def stream(self, msg: str, frame_id: int=None):
        self._stream_counter += 1
        formatted_msg = f'[blue]≈ {msg}[/blue]'
        if frame_id is not None:
            formatted_msg = f'[blue]≈ [Frame {frame_id}] {msg}[/blue]'
        self._logger.log(LEVEL_STREAM, formatted_msg, extra={'markup': True})
    def log_anomaly_report(self, report: dict):
        frame_id = report.get('frame_id', 'Unknown')
        confidence = report.get('confidence', 0.0)
        causes = report.get('top_causes', [])
        table = Table(title=f'Anomaly Report | Frame: {frame_id} | Confidence: {confidence:.1%}', style='red')
        table.add_column('Rank', justify='center', style='cyan')
        table.add_column('Causal Node (Root Cause)', justify='left', style='magenta')
        table.add_column('Contribution', justify='right', style='green')
        for idx, cause_data in enumerate(causes, 1):
            cause_name = cause_data.get('cause', 'Unknown')
            percentage = cause_data.get('percentage', 0.0)
            table.add_row(str(idx), cause_name, f'{percentage:.1%}')
        panel = Panel(table, border_style='red', title='[bold red]CRITICAL ANOMALY DETECTED[/bold red]', expand=False)
        self.console.print(panel)
        self._logger.info(f'Anomaly Report | Frame {frame_id} | Confidence: {confidence:.1%}')
        for idx, cause_data in enumerate(causes, 1):
            self._logger.info(f'   Rank {idx}: {cause_data.get('cause')} ({cause_data.get('percentage'):.1%})')
_LOGGER_INSTANCE = None
def get_logger() -> CIRCALogger:
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = CIRCALogger()
    return _LOGGER_INSTANCE