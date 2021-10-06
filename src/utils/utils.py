import logging
import random
import warnings
from typing import List, Sequence

import numpy as np
import pytorch_lightning as pl
import rich
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree

REQUIRED_CONFIGS = ['datamodule', 'task', 'model.backbone', 'model.header', 'loss', 'optimizer', 'trainer', 'train',
                    'test']


def get_logger(name=__name__, level=logging.INFO):
    """Initializes python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger()


def check_config(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file.
        - check for required configs in the main config
        - disabling warnings
        - easier access to debug mode
        - forcing debug friendly configuration
        - forcing multi-gpu friendly configuration
    Args:
        config (DictConfig): [description]
    """

    # check if required configs are in the main config file
    for cf in REQUIRED_CONFIGS:
        _check_if_in_config(config=config, name=cf)

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.disable_warnings=True>
    if config.get("disable_warnings"):
        log.info(f"Disabling python warnings! <config.disable_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    if config.trainer.get("accelerator") in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info("Forcing ddp friendly configuration! <config.trainer.accelerator=ddp>")
        if "plugins" not in config:
            config['plugins'] = DictConfig(
                {'ddp_plugin': {'_target_': 'pytorch_lightning.plugins.DDPPlugin', 'find_unused_parameters': False}})
        if "ddp_plugin" not in config.plugins:
            config["plugins"].append(
                {'ddp_plugin': {'_target_': 'pytorch_lightning.plugins.DDPPlugin', 'find_unused_parameters': False}})

    if config.trainer.get("accelerator") == 'ddp_cpu' and config.trainer.precision == 16:
        log.warning(f'You are using ddp_cpu without precision=16. This can lead to a crash! Use 64 or 32!')

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)
    else:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
        config['seed'] = seed
        log.info(f"No seed specified! Seed set to {seed}")
    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


def _check_if_in_config(config: DictConfig, name: str):
    name_parts = name.split('.')
    for part in name_parts:
        if part in config:
            config = config.get(part)
        else:
            raise ValueError(f'You need to define a value for ({name}) else the system will not start!')


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "task",
                "model",
                "optimizer",
                "datamodule",
                "callbacks",
                "metric",
                "logger",
                "seed",
                "train",
                "test"
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        task: pl.LightningModule,
        model: pl.LightningModule,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    """

    hparams = {"trainer": config["trainer"], "task": config["task"], "model": config["model"],
               "datamodule": config["datamodule"], 'loss': config['loss'], 'optimizer': config['optimizer']}

    # choose which parts of hydra config will be saved to loggers
    if "optimizer" in config:
        hparams["optimizer"] = config["optimizer"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = empty


def finish(
        config: DictConfig,
        task: pl.LightningModule,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()
