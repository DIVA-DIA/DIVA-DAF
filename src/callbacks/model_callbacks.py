import logging
import os
import sys
import traceback
from typing import Optional, Callable, Dict, Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, BaseFinetuning
from pytorch_lightning.callbacks.finetuning import multiplicative
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim import Optimizer

log = logging.getLogger(__name__)


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
        self.CHECKPOINT_NAME_LAST = 'task_last'

    @rank_zero_only
    def _del_model(self, trainer: pl.Trainer, filepath: str) -> None:
        if trainer.should_rank_save_checkpoint and self._fs.exists(filepath):
            parent_dir = self._fs._parent(filepath)
            # delete all files in directory
            for path in self._fs.ls(parent_dir):
                if self._fs.exists(path):
                    self._fs.rm(path)
            # delete directory
            self._fs.rmdir(parent_dir)
            log.debug(f"Removed checkpoint: {filepath}")

    def _save_model(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_model(trainer=trainer, filepath=filepath)
        if not trainer.is_global_zero:
            return

        model = trainer.lightning_module.model
        metric_candidates = self._monitor_candidates(trainer, epoch=trainer.current_epoch, step=trainer.global_step)
        # check if it is a last save or not
        if 'last' not in filepath:
            # fixed pathing problem
            format_backbone_filename = self._format_checkpoint_name(filename=self.backbone_filename,
                                                                    metrics=metric_candidates)
            format_header_filename = self._format_checkpoint_name(filename=self.header_filename,
                                                                  metrics=metric_candidates)
        else:
            format_backbone_filename = self.backbone_filename.split('/')[-1] + '_last'
            format_header_filename = self.header_filename.split('/')[-1] + '_last'

        torch.save(model.backbone.state_dict(), os.path.join(self.dirpath, format_backbone_filename + '.pth'))
        torch.save(model.header.state_dict(), os.path.join(self.dirpath, format_header_filename + '.pth'))


class CheckBackboneHeaderCompatibility(Callback):

    def __init__(self):
        self.checked = False

    @rank_zero_only
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if self.checked:
            return
        # get the datamodule and the dim of the input
        dim = (trainer.datamodule.batch_size, *trainer.datamodule.dims)
        # test if backbone works
        try:
            b_output = pl_module.model.backbone(torch.rand(*dim, device=pl_module.device))
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


class FreezeBackboneOrHeader(BaseFinetuning):
    r"""

    Finetune a backbone or header model based on a learning rate user-defined scheduling.
    When the backbone or header learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:

        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.

        lambda_func: Scheduling function for increasing backbone learning rate.

        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model

        backbone_initial_lr: Optional, Inital learning rate for the backbone.
            By default, we will use current_learning /  backbone_initial_ratio_lr

        should_align: Wheter to align with current learning rate when backbone learning
            reaches it.

        initial_denom_lr: When unfreezing the backbone, the intial learning rate will
            current_learning_rate /  initial_denom_lr.

        train_bn: Wheter to make Batch Normalization trainable.

        verbose: Display current learning rate for model and backbone

        round: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """

    def __init__(
            self,
            unfreeze_backbone_at_epoch: int = -1,
            lambda_func: Callable = multiplicative,
            backbone_initial_ratio_lr: float = 10e-2,
            backbone_initial_lr: Optional[float] = None,
            should_align: bool = True,
            initial_denom_lr: float = 10.0,
            train_bn: bool = True,
            verbose: bool = False,
            round: int = 12,
    ):
        super().__init__()

        self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
        self.lambda_func: Callable = lambda_func
        self.backbone_initial_ratio_lr: float = backbone_initial_ratio_lr
        self.backbone_initial_lr: Optional[float] = backbone_initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.round: int = round
        self.previous_backbone_lr: Optional[float] = None

    def on_save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[int, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }

    def on_load_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
            callback_state: Dict[int, List[Dict[str, Any]]]
    ) -> None:
        self.previous_backbone_lr = callback_state["previous_backbone_lr"]
        super().on_load_checkpoint(trainer, pl_module.model, callback_state["internal_optimizer_metadata"])

    def on_fit_start(self, trainer, pl_module):
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module.model, "backbone") and isinstance(pl_module.model.backbone, nn.Module):
            return super().on_fit_start(trainer, pl_module.model)
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: "pl.LightningModule"):
        self.freeze(pl_module.model.backbone)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int):
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                pl_module.model.backbone,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.round)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.round)}"
                )
