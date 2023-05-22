from typing import Union, Optional, OrderedDict

import pytorch_lightning as pl
import torch.nn
from torchvision.models._utils import IntermediateLayerGetter


class BackboneHeaderModel(pl.LightningModule):
    """A generic model class to provide the possibility to create different backbone/header combinations"""

    def __init__(self, backbone: Union[pl.LightningModule, torch.nn.Module],
                 header: Union[pl.LightningModule, torch.nn.Module], backbone_output_layer: Optional[str] = None):
        super().__init__()

        # sanity check if the last layer of the backbone is compatible with the first layer of the header
        if backbone_output_layer is not None:
            return_layer = {backbone_output_layer: 'out'}
            self.backbone = IntermediateLayerGetter(model=backbone, return_layers=return_layer)
        else:
            self.backbone = backbone
        self.header = header

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, OrderedDict):
            x = x['out']
        x = self.header(x)
        return x

