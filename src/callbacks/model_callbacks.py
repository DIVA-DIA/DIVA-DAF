import os
from typing import Dict, Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint


class SaveModelStateDictAndTaskCheckpoint(ModelCheckpoint):
    """
    Saves the neural network weights into a pth file.
    It produces a file for each the encoder and the header.
    """

    def __init__(self, backbone_filename: Optional[str] = 'backbone', header_filename: Optional[str] = 'header',
                 **kwargs):
        super(SaveModelStateDictAndTaskCheckpoint, self).__init__(**kwargs)
        self.backbone_filename = backbone_filename
        self.header_filename = header_filename

    def on_save_checkpoint(self, trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        # get the generic model from the pl_module
        model: pl.LightningModule = pl_module.model
        # replace the epoch variable in the file names
        # save the encoder and the decoder with torch.save() in self.dirpath
        torch.save(model.backbone.state_dict(), os.path.join(self.dirpath, self.backbone_filename + '.pth'))
        torch.save(model.header.state_dict(), os.path.join(self.dirpath, self.header_filename + '.pth'))
        return super().on_save_checkpoint(trainer=trainer, pl_module=pl_module, checkpoint=checkpoint)


