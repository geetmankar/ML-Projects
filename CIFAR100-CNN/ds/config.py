from dataclasses import dataclass


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


@dataclass
class CIFAR100Config:
    paths: Paths
    params: Params