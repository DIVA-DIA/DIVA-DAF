import logging
import os
import sys
import traceback
from typing import Optional, OrderedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem

log = logging.getLogger(__name__)


class SaveModelStateDictAndTaskCheckpoint(ModelCheckpoint):
    """
    Saves the neural network weights into a pth file.
    It produces a file for each the encoder and the header.
    The encoder file is named after the backbone_filename and the header file after the header_filename.
    The backbone_filename and the header_filename can be specified in the constructor.
    """

    def __init__(self, backbone_filename: Optional[str] = 'backbone', header_filename: Optional[str] = 'header',
                 **kwargs):
        """
        :param backbone_filename: Filename of the backbone checkpoint
        :param header_filename: Filename of the header checkpoint
        """
        super(SaveModelStateDictAndTaskCheckpoint, self).__init__(**kwargs)
        self.backbone_filename = backbone_filename
        self.header_filename = header_filename
        self.CHECKPOINT_NAME_LAST = 'task_last'

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """
        Saves the header and backbone state dict and the task checkpoint in a combined folder.
        The name of the folder ist the epoch the checkpoint was created on.

        The task checkpoint is saved as task_last.pth if it is the last checkpoint and as task_epoch=x.pth otherwise.
        :param trainer: The trainer object
        :param filepath: The filepath of the model state dict
        """
        if not trainer.is_global_zero:
            return
        super()._save_checkpoint(trainer=trainer, filepath=filepath)

        model = trainer.lightning_module.model
        metric_candidates = self._monitor_candidates(trainer)
        # check if it is a last save or not
        if 'last' not in filepath:
            # fixed pathing problem
            format_backbone_filename = self._format_checkpoint_name(filename=self.backbone_filename,
                                                                    metrics=metric_candidates)
            format_header_filename = self._format_checkpoint_name(filename=self.header_filename,
                                                                  metrics=metric_candidates)
            if trainer.current_epoch > 0:
                # remove the last checkpoint folder
                filepath_old = os.path.join(self.dirpath, f"epoch={trainer.current_epoch - 1}")
                self._del_old_folder(filepath_old, trainer)
        else:
            format_backbone_filename = self.backbone_filename.split('/')[-1] + '_last'
            format_header_filename = self.header_filename.split('/')[-1] + '_last'

        torch.save(model.backbone.state_dict(), os.path.join(self.dirpath, format_backbone_filename + '.pth'))
        torch.save(model.header.state_dict(), os.path.join(self.dirpath, format_header_filename + '.pth'))

    @rank_zero_only
    def _del_old_folder(self, filepath: str, trainer: "pl.Trainer") -> None:
        """
        Deletes the old folder of the last checkpoint if the current epoch is better based on the monitor metric.

        :param filepath: The filepath of the old folder
        :param trainer: The trainer object
        """
        file_system = get_filesystem(filepath)
        if file_system.exists(filepath):
            # delete all files in directory
            for path in file_system.ls(filepath):
                if file_system.exists(path):
                    trainer.strategy.remove_checkpoint(path)
            # delete directory
            file_system.rm(filepath, recursive=True)
            log.debug(f"Removed checkpoint: {filepath}")


class CheckBackboneHeaderCompatibility(Callback):
    """
    Checks if the backbone and the header are compatible.
    This is checked by passing a random tensor through the backbone and the header.
    If the backbone and the header are not compatible, the program is terminated.

    """

    def __init__(self):
        self.checked = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if self.checked:
            return
        # get the datamodule and the dim of the input
        dim = (trainer.datamodule.batch_size, *trainer.datamodule.dims)
        # test if backbone works
        try:
            b_output = pl_module.model.backbone(torch.rand(*dim, device=pl_module.device))
            if isinstance(b_output, OrderedDict):
                b_output = b_output['out']
            log.info(f"Backbone has an output of {b_output.shape}")
        except RuntimeError as e:
            log.error(f"Problem in the backbone! Your image dimension is {trainer.datamodule.dims}")
            log.error(e)
            log.error(traceback.format_exc())
            sys.exit(1)
        # test if backbone matches header
        try:
            pl_module(torch.rand(*dim, device=pl_module.device))
        except RuntimeError as e:
            log.error(f'Backbone and Header are not fitting together! Backbone output dimensions {b_output.shape}.'
                      f'Perhaps flatten header input first.')
            log.error(f'Output size (first dimension = batch size) of the backbone flattened:'
                      f' {torch.nn.Flatten()(b_output).shape}')
            log.error(e)
            log.error(traceback.format_exc())
            sys.exit(1)

        self.checked = True
