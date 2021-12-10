import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer, plugins
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from src.models.backbone_header_model import BackboneHeaderModel
from src.utils import utils

log = utils.get_logger(__name__)


def execute(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Write current run dir into outputs/run_dir_paths.txt
    _write_current_run_dir(config)

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
    model: BackboneHeaderModel = BackboneHeaderModel(backbone=backbone, header=header)

    # Init optimizer
    log.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters(recurse=True))

    log.info(f"Instantiating loss<{config.loss._target_}>")
    loss: torch.nn.Module = hydra.utils.instantiate(config.loss)

    metric_train = None
    metric_val = None
    metric_test = None
    if 'metric' in config:
        log.info(f"Instantiating metric<{config.metric._target_}>")
        metric_train = hydra.utils.instantiate(config.metric)
        metric_val = hydra.utils.instantiate(config.metric)
        metric_test = hydra.utils.instantiate(config.metric)

    # Init the task as lightning module
    log.info(f"Instantiating model <{config.task._target_}>")
    task: LightningModule = hydra.utils.instantiate(config.task,
                                                    model=model,
                                                    optimizer=optimizer,
                                                    loss_fn=loss,
                                                    metric_train=metric_train,
                                                    metric_val=metric_val,
                                                    metric_test=metric_test,
                                                    )

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
                logger.append(hydra.utils.instantiate(lg_conf))

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
    utils.log_hyperparameters(
        config=config,
        model=model,
        trainer=trainer,
    )

    if config.save_config:
        RUN_CONFIG_NAME = 'run_config.yaml'
        log.info("Saving the current config into the output directory!")
        # cwd is already the output directory so we dont need a full path
        if trainer.is_global_zero:
            with open(RUN_CONFIG_NAME, mode='w') as fp:
                OmegaConf.set_struct(config, False)
                config['hydra'] = HydraConfig.instance().cfg['hydra']
                OmegaConf.set_struct(config, True)
                OmegaConf.save(config=config, f=fp)
            if config.get('logger') is not None and 'wandb' in config.get('logger'):
                if '_target_' in config.logger.wandb:
                    run_config_folder_path = Path(wandb.run.dir) / 'run_config'
                    run_config_folder_path.mkdir(exist_ok=True)
                    shutil.copyfile(RUN_CONFIG_NAME, str(run_config_folder_path / RUN_CONFIG_NAME))

    if config.train:
        # Train the model
        log.info("Starting training!")
        trainer.fit(model=task, datamodule=datamodule)

    # Evaluate model on test set after training
    if config.test:
        log.info("Starting testing!")
        results = trainer.test(model=task, datamodule=datamodule)
        log.info(f'Test output: {results}')

    if config.predict:
        log.info("Starting prediction!")
        trainer.predict(model=task, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        task=task,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    _print_best_paths(conf=config, trainer=trainer)

    _print_run_command(trainer=trainer)

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

    freeze = False
    strict = True
    if 'strict' in config.model.get(part_name):
        log.info(f"The model part {part_name} will be loaded with strict={config.model.get(part_name).strict}")
        strict = config.model.get(part_name).strict
        del config.model.get(part_name).strict

    if 'freeze' in config.model.get(part_name):
        log.info(f"The model part {part_name} is frozen during all stages!")
        freeze = True
        del config.model.get(part_name).freeze

    if "path_to_weights" in config.model.get(part_name):
        log.info(f"Loading {part_name} weights from <{config.model.get(part_name).path_to_weights}>")
        path_to_weights = config.model.get(part_name).path_to_weights
        del config.model.get(part_name).path_to_weights
        part: LightningModule = hydra.utils.instantiate(config.model.get(part_name))
        missing_keys, unexpected_keys = part.load_state_dict(torch.load(path_to_weights, map_location='cpu'), strict=strict)
        if missing_keys:
            log.warning(f"When loading the model part {part_name} these keys where missed: \n {missing_keys}")
        if unexpected_keys:
            log.warning(f"When loading the model part {part_name} these keys where to much: \n {unexpected_keys}")
    else:
        if config.test and not config.train:
            log.warning(f"You are just testing without a trained {part_name} model! "
                        "Use 'path_to_weights' in your model to load a trained model")
        if config.predict and not config.train:
            log.warning(f"You are just predicting without a trained {part_name} model! "
                        "Use 'path_to_weights' in your model to load a trained model")
        part: LightningModule = hydra.utils.instantiate(config.model.get(part_name))

    if freeze:
        for param in part.parameters():
            param.requires_grad = False

        part.eval()

    return part


def _print_best_paths(conf: DictConfig, trainer: Trainer):
    """
    Print out the best checkpoint paths for the task, the backbone, and the header.

    Args:
        conf: the hydra config
        trainer: the current pl trainer
    """
    if not conf.train or 'model_checkpoint' not in conf.callbacks:
        return

    def _create_print_path(folder_path: Path, config_file_name: str):
        return folder_path / (Path(config_file_name).name + '.pth')

    # Print path to best checkpoint
    base_path = Path(trainer.checkpoint_callback.best_model_path).parent
    log.info(
        f"Best task checkpoint path:"
        f"\n{trainer.checkpoint_callback.best_model_path}")
    if '_target_' in conf.callbacks.get('model_checkpoint'):
        log.info(
            f"Best backbone checkpoint path:"
            f"\n{_create_print_path(base_path, conf.callbacks.model_checkpoint.backbone_filename)}")
        log.info(
            f"Best header checkpoint path:"
            f"\n{_create_print_path(base_path, conf.callbacks.model_checkpoint.header_filename)}")


def _print_run_command(trainer: Trainer):
    """
    Print out a run command based on the saved run config.

    Args:
        conf: the hydra config
        trainer: the current pl trainer
    """

    run_path = trainer.default_root_dir
    run_config_name = 'run_config.yaml'

    log.info(f'Command to rerun using run_config.yaml:\n'
             f'python run.py -cd="{run_path}" -cn="{run_config_name}"')

    param_str_list = [f'"{p}"' for p in sys.argv[1:]]

    log.info(f'Command to rerun using same command:\n'
             f'python run.py {" ".join(param_str_list)}')


@rank_zero_only
def _write_current_run_dir(config: DictConfig):
    run_dir_log_filename = 'run_dir_log.txt'
    run_dir_log_file = Path(to_absolute_path(config['run_root_dir'])) / run_dir_log_filename
    run_dir_log_file = run_dir_log_file.resolve()
    log.info(f'Writing work dir into run dir log file ({run_dir_log_file})')
    with run_dir_log_file.open('a') as f:
        f.write(f'{os.getcwd()}\n')

