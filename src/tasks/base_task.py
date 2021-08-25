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

from tasks.utils.task_utils import get_callable_dict


class AbstractTask(LightningModule, metaclass=ABCMeta):
    """A general abstract Task.
    Args:
        model: Composed model to use for the task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to ``5e-5``.
        test_output_path: Path relative to the normal output folder where to save the test output
    """

    required_extras: Optional[str] = None

    def __init__(
            self,
            model: Optional[nn.Module] = None,
            loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
            optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
            scheduler_kwargs: Optional[Dict[str, Any]] = None,
            metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
            learning_rate: float = 1e-3,
            test_output_path: Optional[str, Path] = 'output'
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
        self.learning_rate = learning_rate
        self.test_output_path = test_output_path
        self.save_hyperparameters()

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        y_hat = self(x)
        output = {"y_hat": y_hat}
        y_hat = self.to_loss_format(output["y_hat"])
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        y_hat = self.to_metrics_format(output["y_hat"])
        for name, metric in self.metrics.items():
            if isinstance(metric, torchmetrics.metric.Metric):
                metric(y_hat, y)
                logs[name] = metric  # log the metric itself if it is of type Metric
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
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        output = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in output["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True,
                      sync_dist=True)
        return output["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True,
                      sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True,
                      sync_dist=True)

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        optimizer = self.optimizer
        if not isinstance(self.optimizer, Optimizer):
            self.optimizer_kwargs["lr"] = self.learning_rate
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
