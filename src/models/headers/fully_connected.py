import torch
from torch import nn


class FCNHead(nn.Sequential):
    # taken from https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class ResNetHeader(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 109512):
        super(ResNetHeader, self).__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class SingleLinear(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 109512):
        super(SingleLinear, self).__init__()

        self.fc = nn.Sequential(
            torch.nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
