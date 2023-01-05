import torch
from torch import nn


class ConvPoolHeader(nn.Module):
    def __init__(self, in_channels: int = 8, num_conv_channels: int = 32, num_classes: int = 4):
        super(ConvPoolHeader, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_conv_channels, kernel_size=3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(num_features=num_conv_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
            nn.Linear(num_conv_channels, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PoolHeader(nn.Module):
    def __init__(self, in_channels: int = 8, num_classes: int = 4):
        super(PoolHeader, self).__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class UNetFCNHead(nn.Module):
    def __init__(self, num_classes: int, features: int = 64):
        super().__init__()

        self.classifier = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x):
        return self.classifier(x)
