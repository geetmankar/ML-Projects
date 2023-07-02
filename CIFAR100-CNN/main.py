import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
import torchvision.transforms as tt
from torch import nn
from torchvision.models import ResNet34_Weights
from logging import Logger
import hydra
from hydra.core.config_store import ConfigStore

from ds.model import CIFAR100Classifier
from ds.config import CIFAR100Config
from ds.runner import Runner, run_epoch
from ds.dataloader import create_device_dataloader, to_device, get_default_device
from ds.tensorboard import TensorboardExperiment





cs = ConfigStore.instance()
cs.store(name="cifar100_config", node=CIFAR100Config)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: CIFAR100Config):

    TRAIN_DATA = CIFAR100(cfg.paths.data, train=True, transform=tt.ToTensor())
    # TEST_DATA  = CIFAR100(cfg.paths.data, train=False)

    torch.manual_seed(42)
    val_size = int(0.1 * len(TRAIN_DATA))
    train_set, valid_set = random_split(TRAIN_DATA,
                                        [len(TRAIN_DATA)-val_size, val_size])
    
    train_loader = create_device_dataloader(cfg.params.batch_size, train_set,
                                            num_workers=4, shuffle=True)
    
    valid_loader = create_device_dataloader(cfg.params.batch_size, valid_set,
                                            num_workers=4, shuffle=False)
    
    # Main transforma for the training data
    Main_Transforms = nn.Sequential(
            tt.RandomHorizontalFlip(),
            tt.RandomAutocontrast(1),
            tt.RandomRotation(10),
            ResNet34_Weights.IMAGENET1K_V1.transforms(),
            )

    # Model, Optimizer, and Learning-Rate Scheduler
    model = to_device(CIFAR100Classifier(transforms=Main_Transforms),
                      get_default_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.max_lr,
                                 weight_decay=cfg.params.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                            max_lr=cfg.params.max_lr,
                            epochs=cfg.params.epoch_count,
                            steps_per_epoch=len(train_loader))
    

    # Create the runners
    train_runner = Runner(train_loader, model, optimizer, lr_scheduler)
    valid_runner = Runner(valid_loader, model)

    # Setup the Experiment Tracker
    tracker = TensorboardExperiment(log_path=cfg.paths.log)


    # Run the epochs
    for epoch_id in range(cfg.params.epoch_count):
        run_epoch(train_runner, valid_runner, tracker, epoch_id)

        # Average epoch metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{cfg.params.epoch_count}]",
                f"Test Accuracy: {valid_runner.avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",

            ]
        )

        Logger.info("\n" + summary + "\n")


        # Reset the runners
        train_runner.reset()
        valid_runner.reset()


        # Flush the tracker for live updates
        tracker.flush()



if __name__=='__main__':
    main()