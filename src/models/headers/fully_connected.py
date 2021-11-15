import torch
from torch import nn


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
