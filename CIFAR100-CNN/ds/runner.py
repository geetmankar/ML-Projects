from typing import Any, Optional

import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from ds.metrics import Metric
from ds.tracking import Tracker, Stage
from ds.dataloader import DeviceDataLoader


class Runner:
    def __init__(
        self,
        loader: DeviceDataLoader,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.OneCycleLR] = None,
    ) -> None:
        self.run_count = 0
        self.loader = loader
        self.accuracy_metric = Metric()
        self.lr_metric = Metric()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # Loss function
        self.compute_loss = nn.CrossEntropyLoss(reduction="mean")
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        # Assume Stage based on presence of optimizer
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def run(self, desc: str, experiment: Tracker):
        self.model.train(self.stage is Stage.TRAIN)

        for batch in tqdm(self.loader, desc=desc, ncols=80):
            loss, batch_accuracy = self._run_single(batch)

            experiment.add_batch_metric("accuracy", batch_accuracy, self.run_count)

            if self.optimizer:
                # Reverse-mode AutoDiff (backpropagation)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                experiment.add_batch_metric("lr", self.get_lr(self.optimizer),
                                            self.run_count)
                self.lr_scheduler.step()

    def _run_single(self, batch: Any):
        self.run_count += 1
        imgs, labels = batch
        batch_size: int = imgs.shape[0]
        prediction = self.model(imgs)
        loss = self.compute_loss(prediction, labels)

        # Compute Batch Validation Metrics
        y_ = labels.detach()
        y_prediction = torch.argmax(prediction.detach(), axis=1)
        batch_accuracy: float = accuracy_score(y_, y_prediction)
        self.accuracy_metric.update(batch_accuracy, batch_size)
        self.lr_metric.update(self.get_lr(self.optimizer), batch_size)

        self.y_true_batches += [y_]
        self.y_pred_batches += [y_prediction]
        return loss, batch_accuracy

    def reset(self):
        torch.cuda.empty_cache()
        self.accuracy_metric = Metric()
        self.lr_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []


def run_epoch(
    train_runner: Runner,
    test_runner: Runner,
    experiment: Tracker,
    epoch_id: int,
):
    # Training Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment)

    # Log Training Epoch Metrics
    experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch_id)

    # Testing Loop
    experiment.set_stage(Stage.VAL)
    test_runner.run("Validation Batches", experiment)

    # Log Validation Epoch Metrics
    experiment.add_epoch_metric("accuracy", test_runner.avg_accuracy, epoch_id)
    experiment.add_epoch_confusion_matrix(
        test_runner.y_true_batches, test_runner.y_pred_batches, epoch_id
    )