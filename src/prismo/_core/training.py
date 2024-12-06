import numpy as np


class EarlyStopper:
    """Class to manage early stopping of model training.

    Adapted from https://gist.github.com/stefanonardo.
    """

    def __init__(self, mode: str = "min", min_delta: float = 0.0, patience: int = 10, percentage: bool = False):
        """Initialize the EarlyStopper.

        Args:
            mode: The mode to use for comparison (min or max).
            min_delta: The minimum change in the monitored metric to be considered an improvement.
            patience: The number of epochs to wait before stopping.
            percentage: Whether the minimum delta is a percentage of the current value.
        """
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best: float | None = None
        self.num_bad_epochs: int = 0
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda metrics: False

    def step(self, metrics: float) -> bool:
        """Determine if the training should stop.

        Args:
            metrics: The current value of the metric being monitored.
        """
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if np.isinf(metrics):
            self.num_bad_epochs += 1
        elif self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

    def _init_is_better(self, mode: str, min_delta: float, percentage: bool) -> None:
        """Initialize the comparator based on the mode.

        Args:
            mode: The mode to use for comparison (min or max)
            min_delta: The minimum change in the monitored metric to be considered an improvement.
            percentage: Whether the minimum delta is a percentage of the current value.
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            delta = min_delta / 100
            if mode == "min":
                self.is_better = lambda a, best: a < best - abs(best) * delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + abs(best) * delta
