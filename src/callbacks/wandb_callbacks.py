from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src.utils import utils


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """
    Get WandbLogger from trainer or loggers.

    Args:
        trainer: PyTorch Lightning trainer.

    Returns:
        WandbLogger

    Raises:
        ValueError: If WandbLogger was not found
    """
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            return logger

    raise ValueError(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModelWithWandb(Callback):
    """
    Make WandbLogger watch model at the beginning of the run.

    Args:
        log_category: Category of the model to log ("gradients", "parameters", "all", or None).
        log_freq: How often to log the model.
    """

    def __init__(self, log_category: str = "gradients", log_freq: int = 100):
        self.log_category = log_category
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            logger = get_wandb_logger(trainer=trainer)
            logger.watch(model=pl_module.model, log=self.log_category, log_freq=self.log_freq)
        except ValueError as e:
            logger = utils.get_logger(__name__)
            logger.error('No wandb logger found. WatchModelWithWandb callback will not do anything.')
