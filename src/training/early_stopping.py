"""Small, serializable early-stopping state shared by training loops."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    """Track a metric that is maximized; ``patience=0`` disables stopping."""

    patience: int = 2
    min_delta: float = 1e-4
    best: float = float("-inf")
    best_epoch: int = 0
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("early-stopping patience must be non-negative")
        if self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("early-stopping min_delta must be finite and non-negative")

    def update(self, metric: float, epoch: int) -> tuple[bool, bool]:
        if not math.isfinite(metric):
            raise FloatingPointError(f"Non-finite early-stopping metric at epoch={epoch}")
        improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, self.stopped

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "stopped": self.stopped,
        }
