import glob
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchmetrics
import wandb
from matplotlib.patches import Rectangle
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from sklearn.metrics import f1_score, precision_score, recall_score

from src.tasks.utils.outputs import OutputKeys


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class LogConfusionMatrixToWandbVal(Callback):
    """Generate confusion matrix every epoch and send it to wandb during validation.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs[OutputKeys.PREDICTION].detach().cpu().numpy())
            self.targets.append(outputs[OutputKeys.TARGET].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix and upload it with the numbers or as image"""
        if not self.ready:
            return

        if trainer.is_global_zero:
            _create_and_save_conf_mat(trainer=trainer, input_preds=self.preds, input_targets=self.targets, phase='val')
            self.preds.clear()
            self.targets.clear()


class LogConfusionMatrixToWandbTest(Callback):
    """Generate confusion matrix every epoch and send it to wandb during test.
    Expects test step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs[OutputKeys.PREDICTION].detach().cpu().numpy())
            self.targets.append(outputs[OutputKeys.TARGET].detach().cpu().numpy())

    def on_test_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix and upload it with the numbers or as image"""
        if not self.ready:
            return

        _create_and_save_conf_mat(trainer=trainer, input_preds=self.preds, input_targets=self.targets, phase='test')

        self.preds.clear()
        self.targets.clear()


def _create_and_save_conf_mat(trainer, input_preds, input_targets, phase):
    """
    This function creates a confusion matrix and saves it as a tsv file as well as uploading it to wandb as tsv and
    as an image.
    :param trainer:
    :param input_preds:
    :param input_targets:
    :param phase:
    """
    # TODO get from task the preprocess function
    conf_mat_name = f'CM_epoch_{trainer.current_epoch}'
    logger = get_wandb_logger(trainer)
    experiment = logger.experiment

    preds = []
    for step_pred, step_target in zip(input_preds, input_targets):
        preds.append(trainer.model.module.module.to_metrics_format(np.array(step_pred)))

    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(np.array(input_targets)).flatten()

    # only works if preds and targets contains index of class (starting with 0)
    num_classes = max(np.max(preds), np.max(targets)) + 1

    confusion_matrix = torchmetrics.functional.confusion_matrix(target=torch.tensor(targets), preds=torch.tensor(preds),
                                                                num_classes=num_classes)

    # set figure size
    plt.figure(figsize=(14, 8))
    # set labels size
    sn.set(font_scale=1.4)
    # set font size
    fig = sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

    for i in range(confusion_matrix.shape[0]):
        fig.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=3))
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.title(conf_mat_name)

    conf_mat_path = Path(os.getcwd()) / 'conf_mats' / phase
    conf_mat_path.mkdir(parents=True, exist_ok=True)
    conf_mat_file_path = conf_mat_path / (conf_mat_name + '.txt')
    df = pd.DataFrame(confusion_matrix.detach().cpu().numpy())

    # save as csv or tsv to disc
    df.to_csv(path_or_buf=conf_mat_file_path, sep='\t')
    # save tsv to wandb
    experiment.save(glob_str=str(conf_mat_file_path), base_path=os.getcwd())
    # names should be uniqe or else charts from different experiments in wandb will overlap
    experiment.log({f"confusion_matrix_{phase}_img/ep_{trainer.current_epoch}": wandb.Image(plt)},
                   commit=False)
    # according to wandb docs this should also work but it crashes
    # experiment.log(f{"confusion_matrix/{experiment.name}": plt})
    # reset plot
    plt.clf()


class LogF1PrecRecHeatmapToWandb(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs[OutputKeys.PREDICTION].detach().cpu().numpy())
            self.targets.append(outputs[OutputKeys.TARGET].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if not self.ready:
            return

        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        preds = np.concatenate(self.preds).flatten()
        targets = np.concatenate(self.targets).flatten()
        f1 = f1_score(y_true=targets, y_pred=preds, average=None)
        r = recall_score(y_true=targets, y_pred=preds, average=None)
        p = precision_score(y_true=targets, y_pred=preds, average=None)
        data = [f1, p, r]

        # set figure size
        plt.figure(figsize=(14, 3))

        # set labels size
        sn.set(font_scale=1.2)

        # set font size
        sn.heatmap(
            data,
            annot=True,
            annot_kws={"size": 10},
            fmt=".3f",
            yticklabels=["F1", "Precision", "Recall"],
        )

        # names should be uniqe or else charts from different experiments in wandb will overlap
        experiment.log({f"f1_p_r_heatmap/{experiment.name}_ep{trainer.current_epoch}": wandb.Image(plt)}, commit=False)

        # reset plot
        plt.clf()

        self.preds.clear()
        self.targets.clear()
