from typing import Union

import pytorch_lightning as pl
import torch.nn


class EncoderHeaderModel(pl.LightningModule):
    """A generic model class to provide the possibility to create different backbone/header combinations"""

    def __init__(self, backbone: Union[pl.LightningModule, torch.nn.Module],
                 header: Union[pl.LightningModule, torch.nn.Module]):
        super().__init__()

        # sanity check if the last layer of the backbone is compatible with the first layer of the header

        self.backbone = backbone
        self.header = header

    def forward(self, x):
        x = self.backbone(x)
        x = self.header(x)
        return x

