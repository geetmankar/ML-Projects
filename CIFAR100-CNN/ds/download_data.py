from torchvision.datasets import CIFAR100
import logging


def download_cifar100_dataset():
    root_dir = "./data/"
    CIFAR100(root=root_dir, download=True)
    logging.info('Complete!')


if __name__=='__main__':
    download_cifar100_dataset()
