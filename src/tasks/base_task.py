from abc import ABCMeta
from pathlib import Path
from typing import Optional, Union, Type, Mapping, Sequence, Callable, Dict, Any, Tuple, List

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.tasks.utils.task_utils import get_callable_dict
from src.utils import template_utils

log = template_utils.get_logger(__name__)


class AbstractTask(LightningModule, metaclass=ABCMeta):
    """
    Inspired by
    https://github.com/PyTorchLightning/lightning-flash/blob/2ec52e633bb3679f50dd7e30526885a4547e1851/flash/core/model.py

    A general abstract Task.

    Args:
        model: Composed model to use for the task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
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
            metrics: Optional[Union[torchmetrics.Metric, Callable, Mapping, Sequence, None]] = None,
            lr: float = 1e-3,
            test_output_path: Optional[Union[str, Path]] = 'output'
    ):
        super().__init__()
        if model is not None:
            self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.lr = lr
        self.test_output_path = test_output_path
        self.save_hyperparameters()

    def setup(self, stage: str):
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
                log.warn(f'Last batch will be incomplete. Behavior depends on datamodule.drop_last_batch setting.')

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
        output = {"pred": y_hat}
        y_hat = self.to_loss_format(output["pred"])
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        y_hat = self.to_metrics_format(output["pred"])
        for name, metric in self.metrics.items():
            if isinstance(metric, torchmetrics.Metric):
                if name in metric_kwargs:
                    metric.update(y_hat, y, **metric_kwargs[name])
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
        output["loss"] = list(losses.values())[0]
        output["logs"] = logs
        output["y"] = y
        return output

    def to_loss_format(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
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
        output = self.step(batch, batch_idx, **kwargs)
        for key, value in output["logs"].items():
            self.log(f"train/{key}", value, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return output["loss"]

    def validation_step(self, batch: Any, batch_idx: int, **kwargs) -> None:
        output = self.step(batch, batch_idx, **kwargs)
        for key, value in output["logs"].items():
            self.log(f"val/{key}", value, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return output['pred']

    def test_step(self, batch: Any, batch_idx: int, **kwargs) -> None:
        output = self.step(batch, batch_idx, **kwargs)
        for key, value in output["logs"].items():
            self.log(f"test/{key}", value, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return output['pred']

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
