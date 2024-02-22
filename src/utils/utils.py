import logging
import random
import sys
import warnings
from typing import List, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import rich
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree

REQUIRED_CONFIGS = ['datamodule', 'task', 'model.backbone', 'model.header', 'loss', 'optimizer', 'trainer', 'train',
                    'test']


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """
    Gets the Python logger of the system.

    :param name: name of the logger you want to get defaults to __name__
    :type name: str
    :param level: logging level defaults to logging.INFO
    :return: Python logger
    :rtype: logging.Logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger()


@rank_zero_only
def check_config(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file.
        - check for required configs in the main config
        - disabling warnings
        - easier access to debug mode
        - forcing debug friendly configuration
        - forcing multi-gpu friendly configuration
        - setting seed for random number generators
        - setting up default csv logger

    :param config: the main hydra config
    :type config: DictConfig
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

    if config.trainer.get("accelerator") == 'cpu' and config.trainer.precision == 16:
        log.warning(f'You are using ddp_cpu without precision=16. This can lead to a crash! Use 64 or 32!')

    if config.get('experiment_mode') and not config.get('name'):
        log.info("Experiment mode without specifying a name!")
        sys.exit(1)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)
    else:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
        config['seed'] = seed
        log.info(f"No seed specified! Seed set to {seed}")

    if 'freeze' in config.model.backbone and 'freeze' in config.model.header and config.train:
        if config.model.backbone.freeze and config.model.header.freeze:
            log.error(f"Cannot train with no trainable parameters! Both header and backbone are frozen!")

    if 'csv' not in config.logger:
        config.logger['csv'] = hydra.compose('logger/csv')['logger']['csv']

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


def _check_if_in_config(config: DictConfig, name: str) -> None:
    """
    Check if a key is in the config file.

    :param config: Hydra config
    :type config: DictConfig
    :param name: name of the key
    :type name: str
    :raises ValueError: if the key is not in the config file
    """
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
                "loss",
                "metric",
                "logger",
                "seed",
                "train",
                "test",
                "predict"
        ),
        resolve: bool = True,
        add_missing_fields: bool = True,
) -> None:
    """
    Prints content of DictConfig using Rich library and its tree structure.

    :param config: Hydra config
    :type config: DictConfig
    :param fields: Determines which main fields from config will be printed and in what order.
    :type fields: Sequence[str]
    :param resolve: Whether to resolve reference fields of DictConfig.
    :type resolve: bool
    :param add_missing_fields: Whether to add missing fields from config to fields.
    :type add_missing_fields: bool
    """

    style = 'default'
    tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

    if add_missing_fields:
        fields = list(fields)
        for key in sorted(config.keys()):
            if key not in fields:
                fields.append(key)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml", background_color=style))

    rich.print(tree)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
) -> None:
    """
    This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionally, saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters

    :param config: Hydra config
    :type config: DictConfig
    :param model: Lightning model
    :type model: pl.LightningModule
    :param trainer: Lightning trainer
    :type trainer: pl.Trainer
    """

    hparams = {"trainer": config["trainer"], "task": config["task"], "model": config["model"],
               "datamodule": config["datamodule"], 'loss': config['loss'], 'optimizer': config['optimizer'],
               "seed": config['seed'], 'callbacks': config['callbacks']}

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


def empty(*args, **kwargs):
    pass


def finish(
        config: DictConfig,
        task: pl.LightningModule,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """
    Makes sure everything closed properly.

    :param config: Hydra config
    :type config: DictConfig
    :param task: Lightning task
    :type task: pl.LightningModule
    :param model: Lightning model
    :type model: pl.LightningModule
    :param datamodule: Lightning datamodule
    :type datamodule: pl.LightningDataModule
    :param trainer: Lightning trainer
    :type trainer: pl.Trainer
    :param callbacks: Lightning callbacks
    :type callbacks: List[pl.Callback]
    :param logger: Lightning logger
    :type logger: List[pl.loggers.LightningLoggerBase]
    """

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()
