from typing import List, Optional

import os
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer, plugins
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

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

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init optimizer
    log.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    if config.optimizer._target_ == 'torch.optim.Adam':
        config.optimizer.betas = tuple([float(i) for i in config.optimizer.betas.split(',')])
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
                model_name = config.model._target_.split('.')[-1]
                datamodule_name = config.datamodule._target_.split('.')[-1]
                post_fix_path = os.getcwd().split('/')[-2:]
                logger.append(hydra.utils.instantiate(lg_conf, name='_'.join(
                    [str(lg_conf.name), task_name, model_name, datamodule_name, '_'.join(post_fix_path)])))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # could be made nice in the same style as callbacks but atm does not matter
    plugin_list = []
    if config.trainer.get("accelerator") in ["ddp", "ddp_spawn", "ddp2"]:
        plugin_list.append(plugins.DDPPlugin(find_unused_parameters=False))
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

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for Optuna optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
