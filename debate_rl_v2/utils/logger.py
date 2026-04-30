"""TensorBoard logging wrapper with scalar, histogram, and text support."""

from __future__ import annotations

import os
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Thin wrapper around TensorBoard SummaryWriter."""

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        self.enabled = enabled
        if enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    # -- scalars ---------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        if self.writer:
            self.writer.add_scalars(main_tag, values, step)

    # -- histograms ------------------------------------------------------------

    def log_histogram(self, tag: str, values, step: int) -> None:
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    # -- text ------------------------------------------------------------------

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.writer:
            self.writer.add_text(tag, text, step)

    # -- lifecycle -------------------------------------------------------------

    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
