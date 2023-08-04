from enum import Enum, auto
from typing import Protocol


import numpy as np


class Stage(Enum):
    TRAIN = auto()
    VAL = auto()


class Tracker(Protocol):
    def set_stage(self, stage: Stage):
        """Sets the current stage of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(
        self, y_true: list[np.array], y_pred: list[np.array], step: int
    ):
        """Implements logging a confusion matrix at epoch-level."""

    def add_final_confusion_matrix(
        self, y_true: list[np.array], y_pred: list[np.array]
    ):
        """Adds the final confusion matrix for the test (not val) set"""
