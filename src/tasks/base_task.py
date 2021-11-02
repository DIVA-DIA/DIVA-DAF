import os
from abc import ABCMeta
from pathlib import Path
from typing import Optional, Union, Type, Mapping, Sequence, Callable, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchmetrics
import wandb
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.callbacks.wandb_callbacks import get_wandb_logger
from src.tasks.utils.outputs import OutputKeys
from src.tasks.utils.task_utils import get_callable_dict
from src.utils import utils

log = utils.get_logger(__name__)


class AbstractTask(LightningModule, metaclass=ABCMeta):
    """
    Inspired by
    https://github.com/PyTorchLightning/lightning-flash/blob/2ec52e633bb3679f50dd7e30526885a4547e1851/flash/core/model.py

    A general abstract Task.

    Args:
        model: Composed model to use for the task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.Adam`.
        metric_train: Metrics to compute for training.
        metric_val: Metrics to compute for evaluation.
        metric_test: Metrics to compute for testing.
        lr: Learning rate to use for training, defaults to ``5e-5``.
        test_output_path: Path relative to the normal output folder where to save the test output
    """

    def __init__(
            self,
            model: Optional[nn.Module] = None,
            loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
            optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
            scheduler_kwargs: Optional[Dict[str, Any]] = None,
            metric_train: Optional[torchmetrics.Metric] = None,
            metric_val: Optional[torchmetrics.Metric] = None,
            metric_test: Optional[torchmetrics.Metric] = None,
            confusion_matrix_val: Optional[bool] = False,
            confusion_matrix_test: Optional[bool] = False,
            confusion_matrix_log_every_n_epoch: Optional[int] = 1,
            lr: float = 1e-3,
            test_output_path: Optional[Union[str, Path]] = 'predictions'
    ):
        super().__init__()

        resolver_name = 'task'
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(
                resolver_name,
                lambda name: getattr(self, name),
                use_cache=False
            )

        if model is not None:
            self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.metric_train = nn.ModuleDict({} if metric_train is None else get_callable_dict(metric_train))
        self.metric_val = nn.ModuleDict({} if metric_val is None else get_callable_dict(metric_val))
        self.metric_test = nn.ModuleDict({} if metric_test is None else get_callable_dict(metric_test))
        self.confusion_matrix_val = confusion_matrix_val
        self.confusion_matrix_test = confusion_matrix_test
        self.confusion_matrix_log_every_n_epoch = confusion_matrix_log_every_n_epoch
        self.lr = lr
        self.test_output_path = Path(test_output_path)
        self.save_hyperparameters()

    def setup(self, stage: str):
        if self.confusion_matrix_val:
            self.metric_conf_mat_val = torchmetrics.ConfusionMatrix(num_classes=self.trainer.datamodule.num_classes,
                                                                    compute_on_step=False)
        if self.confusion_matrix_test:
            self.metric_conf_mat_test = torchmetrics.ConfusionMatrix(num_classes=self.trainer.datamodule.num_classes,
                                                                     compute_on_step=False)
        if self.trainer.distributed_backend == 'ddp':
            batch_size = self.trainer.datamodule.batch_size
            if stage == 'fit':
                num_samples = len(self.trainer.datamodule.train)
                datasplit_name = 'train'
            elif stage == 'test':
                num_samples = len(self.trainer.datamodule.test)
                datasplit_name = 'test'
            else:
                log.warn(f'Unknown stage ({stage}) during setup!')
                num_samples = -1
                datasplit_name = None

            if num_samples % self.trainer.datamodule.batch_size != 0:
                log.warn(
                    f'Number of sample ({num_samples}) in {datasplit_name} not dividable by batch size ({batch_size}).')
                log.warn(f'Last batch will be incomplete. Behavior depends on datamodule.drop_last setting.')

    def step(self,
             batch: Any,
             batch_idx: int,
             metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
             **kwargs) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        Args:
            batch: the batch with the images (x) and the gt (y) in the order (x, y)
            batch_idx: the index of the given batch
            metric_kwargs: a dictionary with a entry with the additional arguments (pred and y always provided).
                e.g. you have two metrics (A, B) and B takes an additional arguments x and y so the dictionary would
                look like this: {'B': {'x': 'value', 'y': 'value'}}

        """
        if metric_kwargs is None:
            metric_kwargs = {}
        x, y = batch
        y_hat = self(x)
        output = {OutputKeys.PREDICTION: y_hat}
        y_hat = self.to_loss_format(output[OutputKeys.PREDICTION])
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        y_hat = self.to_metrics_format(output[OutputKeys.PREDICTION])
        current_metric = self._get_current_metric()

        for name, metric in current_metric.items():
            if isinstance(metric, torchmetrics.Metric):
                if name in metric_kwargs:
                    metric(y_hat, y, **metric_kwargs[name])
                else:
                    metric(y_hat, y)
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                if name in metric_kwargs:
                    logs[name] = metric(y_hat, y, **metric_kwargs[name])
                else:
                    logs[name] = metric(y_hat, y)
        logs.update(losses)
        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())
            return logs["total_loss"], logs
        output[OutputKeys.LOSS] = list(losses.values())[0]
        output[OutputKeys.LOG] = logs
        output[OutputKeys.TARGET] = y
        return output

    @staticmethod
    def to_loss_format(x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x

    @staticmethod
    def to_metrics_format(x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x

    def forward(self, x: Any) -> Any:
        # some frameworks like torchvision returns dict
        x = self.model(x)

        # this looks like a bug in pycharm
        if torch.jit.isinstance(x, Dict[str, torch.Tensor]):
            out = x['out']
        elif torch.is_tensor(x):
            out = x
        else:
            raise NotImplementedError(f"Unsupported network output type: {type(x)}")
        return out

    def training_step(self, batch: Any, batch_idx: int, **kwargs) -> Any:
        output = self.step(batch=batch, batch_idx=batch_idx, **kwargs)
        for key, value in output[OutputKeys.LOG].items():
            self.log(f"train/{key}", value, on_epoch=True, on_step=True, sync_dist=True, rank_zero_only=True)
        return output

    def validation_step(self, batch: Any, batch_idx: int, **kwargs) -> None:
        output = self.step(batch=batch, batch_idx=batch_idx, **kwargs)
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING \
                and self.confusion_matrix_val \
                and (self.trainer.current_epoch + 1) % self.confusion_matrix_log_every_n_epoch == 0:
            self.metric_conf_mat_val(preds=output[OutputKeys.PREDICTION], target=output[OutputKeys.TARGET])
        for key, value in output[OutputKeys.LOG].items():
            self.log(f"val/{key}", value, on_epoch=True, on_step=True, sync_dist=True, rank_zero_only=True)
        return output

    def validation_epoch_end(self, outputs: Any) -> None:
        if self.trainer.state.stage == RunningStage.SANITY_CHECKING \
                or not self.confusion_matrix_val \
                or (self.trainer.current_epoch + 1) % self.confusion_matrix_log_every_n_epoch != 0:
            return

        hist = self.metric_conf_mat_val.compute()
        hist = hist.cpu().numpy()

        self._create_conf_mat(matrix=hist, stage='val')

        self.metric_conf_mat_val.reset()

    def test_step(self, batch: Any, batch_idx: int, **kwargs) -> None:
        output = self.step(batch=batch, batch_idx=batch_idx, **kwargs)
        if self.confusion_matrix_test:
            self.metric_conf_mat_test(preds=output[OutputKeys.PREDICTION], target=output[OutputKeys.TARGET])
        for key, value in output[OutputKeys.LOG].items():
            self.log(f"test/{key}", value, on_epoch=True, on_step=True, sync_dist=True, rank_zero_only=True)
        return output

    def test_epoch_end(self, outputs: Any) -> None:
        if not self.confusion_matrix_test:
            return

        hist = self.metric_conf_mat_test.compute()
        hist = hist.cpu().numpy()

        self._create_conf_mat(matrix=hist, stage='test')

        self.metric_conf_mat_test.reset()

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        optimizer = self.optimizer
        if not isinstance(self.optimizer, Optimizer):
            self.optimizer_kwargs["lr"] = self.lr
            optimizer = optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_kwargs)
        if self.scheduler:
            return [optimizer], [self._instantiate_scheduler(optimizer)]
        return optimizer

    def _instantiate_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        scheduler = self.scheduler
        if isinstance(scheduler, _LRScheduler):
            return scheduler
        elif issubclass(scheduler, _LRScheduler):
            return scheduler(optimizer, **self.scheduler_kwargs)
        raise MisconfigurationException(
            "scheduler can be a scheduler, a scheduler type with `scheduler_kwargs` "
            f"or a built-in scheduler in {self.available_schedulers()}"
        )

    def _get_current_metric(self):
        if self.trainer.state.stage == RunningStage.TRAINING:
            return self.metric_train
        if self.trainer.state.stage == RunningStage.VALIDATING:
            return self.metric_val
        if self.trainer.state.stage == RunningStage.TESTING:
            return self.metric_test
        return {}

    def _create_conf_mat(self, matrix: np.ndarray, stage: str = 'val'):
        # verify sum of conf mat entries
        pixels_per_crop = self.trainer.datamodule.dims[1] * self.trainer.datamodule.dims[2]
        num_processes = self.trainer.num_processes
        if stage == 'val':
            dataloader = self.trainer.val_dataloaders[0]
        elif stage == 'test':
            assert not self.trainer.test_dataloaders[0].drop_last
            dataloader = self.trainer.test_dataloaders[0]
        else:
            raise ValueError(f'_create_conf_mat received unexpected stage ({stage})')

        num_samples = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        if not dataloader.drop_last:
            num_samples_in_last_step = num_samples % (batch_size * num_processes)  # == 0, when last step is filled
            num_process_filled = num_samples_in_last_step % num_processes  # == 0, when all evenly filled
            additional_crops = num_processes - num_process_filled if num_process_filled > 0 else 0
            total_samples = num_samples + additional_crops
            expected_sum = total_samples * pixels_per_crop
        else:
            num_steps = len(dataloader)
            expected_sum = num_steps * num_processes * batch_size * pixels_per_crop

        matrix_sum = matrix.sum()
        if not np.isclose(a=expected_sum, b=matrix_sum, rtol=2.5e-7):
            log.warn(f'matrix.sum() is not close to expected_sum '
                     f'({matrix_sum} != {expected_sum}, '
                     f'diff: {matrix_sum - expected_sum}, '
                     f'diff_crops: {(matrix_sum - expected_sum) / pixels_per_crop})')

        # print(f'With all_gather: {str(len(outputs[0][OutputKeys.PREDICTION][0]))}')
        conf_mat_name = f'CM_epoch_{self.trainer.current_epoch}'
        logger = get_wandb_logger(self.trainer)
        experiment = logger.experiment

        # set figure size
        plt.figure(figsize=(14, 8))
        # set labels size
        sn.set(font_scale=1.4)
        # set font size
        fig = sn.heatmap(matrix, annot=True, annot_kws={"size": 8}, fmt="g")
        for i in range(matrix.shape[0]):
            fig.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=3))
        plt.xlabel('Predictions')
        plt.ylabel('Targets')
        plt.title(conf_mat_name)
        conf_mat_path = Path(os.getcwd()) / 'conf_mats' / stage
        conf_mat_path.mkdir(parents=True, exist_ok=True)
        conf_mat_file_path = conf_mat_path / (conf_mat_name + '.txt')
        df = pd.DataFrame(matrix)

        # save as csv or tsv to disc
        if self.trainer.is_global_zero:
            df.to_csv(path_or_buf=conf_mat_file_path, sep='\t')
        # save tsv to wandb
        experiment.save(glob_str=str(conf_mat_file_path), base_path=os.getcwd())
        # names should be uniqe or else charts from different experiments in wandb will overlap
        experiment.log({f"confusion_matrix_{stage}_img/ep_{self.trainer.current_epoch}": wandb.Image(plt)},
                       commit=False)
