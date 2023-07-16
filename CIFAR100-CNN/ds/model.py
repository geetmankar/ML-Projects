import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class CIFAR100Classifier(nn.Module):
    def __init__(self, transforms, num_classes:int=100, pretrained=True):
        super().__init__()

        self.ResNet = (resnet34(weights=ResNet34_Weights.DEFAULT) 
                        if pretrained else resnet34())
        self.network = nn.Sequential(
            transforms,
            self.ResNet,
            nn.Linear(self.ResNet.fc.out_features, num_classes),
        )

    def forward(self, xb):
        return self.network(xb)