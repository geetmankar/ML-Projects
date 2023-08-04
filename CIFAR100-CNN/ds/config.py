from dataclasses import dataclass
from typing import Optional


@dataclass
class Paths:
    log: str
    data: str


@dataclass
class Params:
    epoch_count: int
    max_lr: float
    weight_decay: float
    batch_size: int
    grad_clip: Optional[float] = None


@dataclass
class CIFAR100Config:
    paths: Paths
    params: Params
    workers: int
