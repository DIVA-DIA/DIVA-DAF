import os
import random
from typing import List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer, plugins
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src.models.backbone_header_model import BackboneHeaderModel
from src.utils import template_utils

log = template_utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)
    else:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
        log.info(f"No seed specified! Seed set to {seed}")

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init Lightning model backend
    log.info(f"Instantiating backbone model <{config.model.backbone._target_}>")
    backbone = _load_model_part(config=config, part_name='backbone')

    # Init Lightning model header
    log.info(f"Instantiating header model <{config.model.header._target_}>")
    header: LightningModule = _load_model_part(config=config, part_name='header')

    # container model
    model: LightningModule = BackboneHeaderModel(backbone=backbone, header=header)

    # Init optimizer
    log.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters(recurse=True))

    # Init the task as lightning module
    log.info(f"Instantiating model <{config.task._target_}>")
    task: LightningModule = hydra.utils.instantiate(config.task, model=model, optimizer=optimizer)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                task_name = config.task._target_.split('.')[-1]
                backbone_name = config.model.backbone._target_.split('.')[-1]
                header_name = config.model.header._target_.split('.')[-1]
                datamodule_name = config.datamodule._target_.split('.')[-1]
                post_fix_path = os.getcwd().split('/')[-2:]
                logger.append(hydra.utils.instantiate(lg_conf, name='_'.join(
                    [str(lg_conf.name), task_name, backbone_name, header_name, datamodule_name, '_'.join(post_fix_path)])))

    # Init Trainer Plugins
    plugin_list: List[plugins.Plugin] = []
    if "plugins" in config:
        for _, pl_config in config.plugins.items():
            if "_target_" in pl_config:
                log.info(f"Instantiating plugin <{pl_config._target_}>")
                plugin_list.append(hydra.utils.instantiate(pl_config))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, plugins=plugin_list, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    template_utils.log_hyperparameters(
        config=config,
        model=model,
        task=task,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if config.save_config:
        log.info("Saving the current config into the output directory!")
        # cwd is already the output directory so we dont need a full path
        with open('config.yaml', mode='w') as fp:
            OmegaConf.save(config=config, f=fp)

    if config.train:
        # Train the model
        log.info("Starting training!")
        trainer.fit(model=task, datamodule=datamodule)

    # Evaluate model on test set after training
    if config.test:
        log.info("Starting testing!")
        trainer.test(model=task, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    template_utils.finish(
        config=config,
        task=task,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if config.train:
        # Print path to best checkpoint
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for Optuna optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


def _load_model_part(config: DictConfig, part_name: str):
    """
    Checks if a given model part (backbone or header) has a path to a pretrained model and loads this model.
    If there is no pretrained model the model will be initialised randomly.

    :'path_to_weights' in your model config points to the file with the weights to load them.
    :'strict' if you want to load it in a strict fashion. Default is True

    :return
        LightningModule: The loaded network
    """
    missing_keys = []
    unexpected_keys = []

    strict = True
    if 'strict' in config.model.get(part_name):
        log.info(f"The model part {part_name} will be loaded with strict={config.model.get(part_name).strict}")
        strict = config.model.get(part_name).strict
        del config.model.get(part_name).strict

    if "path_to_weights" in config.model.get(part_name):
        log.info(f"Loading {part_name} weights from <{config.model.get(part_name).path_to_weights}>")
        path_to_weights = config.model.get(part_name).path_to_weights
        del config.model.get(part_name).path_to_weights
        part: LightningModule = hydra.utils.instantiate(config.model.get(part_name))
        missing_keys, unexpected_keys = part.load_state_dict(torch.load(path_to_weights), strict=strict)
    else:
        if config.test and not config.train:
            log.warn(f"You are just testing without a trained {part_name} model! "
                     "Use 'path_to_weights' in your model to load a trained model")
        part: LightningModule = hydra.utils.instantiate(config.model.get(part_name))

    if missing_keys is not None:
        log.warn(f"When loading the model part {part_name} these keys where missed: \n {missing_keys}")
    if unexpected_keys is not None:
        log.warn(f"When loading the model part {part_name} these keys where to much: \n {unexpected_keys}")
    return part
