from __future__ import annotations

import os

import numpy as np
import torch
from loguru import logger


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 7, verbose: bool = False):
        """
        Args:
            patience: How long to wait after the last time validation loss improved.
            verbose: If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(
        self, val_loss: float, model: torch.nn.Module, model_dir: str, tuning_metric: str,
    ):
        score = -val_loss if tuning_metric == 'loss' else val_loss
        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir, tuning_metric)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}',
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(
        self, val_loss: float, model: torch.nn.Module, model_dir: str, tuning_metric: str,
    ):
        """Saves model when validation loss decreases or accuracy/f1 increases."""
        if self.verbose:
            metric_message = 'decreased' if tuning_metric == 'loss' else 'increased'
            logger.info(
                f'Validation {tuning_metric} {metric_message} '
                f'({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...',
            )
        model.save_pretrained(model_dir)
        torch.save(
            model.state_dict(), os.path.join(
                model_dir, 'model_state.bin',
            ),
        )
        self.val_loss_min = val_loss

    def _is_improvement(self, score: float) -> bool:
        """Check if the current score is an improvement over the best score."""
        return score > self.best_score if self.best_score is not None else True
