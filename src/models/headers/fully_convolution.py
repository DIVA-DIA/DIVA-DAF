from typing import Tuple, OrderedDict

from torch import nn


class ResNetFCNHead(nn.Sequential):
    """
    FCN header for resnets. The in_channels are fixed for the different resnet architectures:
    resnet18, 34 = 512
    resnet50, 101, 152 = 2048
    """

    def __init__(self, in_channels: int, num_classes: int, output_dims: Tuple[int, int]):
        self.output_dims = output_dims
        if len(self.output_dims) > 2:
            self.output_dims = self.output_dims[-2:]
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=(3, 3), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=inter_channels),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(in_channels=inter_channels, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1)),
        ]

        super(ResNetFCNHead, self).__init__(*layers)

    def forward(self, x):
        x = super(ResNetFCNHead, self).forward(x)
        x = nn.functional.interpolate(x, size=self.output_dims, mode="bilinear", align_corners=False)
        return x
