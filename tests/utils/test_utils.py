import pytest
import io
import sys
from omegaconf import DictConfig

from src.utils.utils import _check_if_in_config, REQUIRED_CONFIGS, check_config, print_config


@pytest.fixture
def get_dict():
    return DictConfig({'plugins': {
        'ddp_plugin': {'_target_': 'pytorch_lightning.plugins.DDPPlugin', 'find_unused_parameters': False}}, 'task': {
        '_target_': 'src.tasks.semantic_segmentation.semantic_segmentation.SemanticSegmentation',
        'confusion_matrix_log_every_n_epoch': 1, 'confusion_matrix_val': True, 'confusion_matrix_test': True},
        'loss': {'_target_': 'torch.nn.CrossEntropyLoss'},
        'metric': {'_target_': 'src.metrics.divahisdb.HisDBIoU',
                   'num_classes': '5'}, 'model': {
            'backbone': {'_target_': 'pl_bolts.models.vision.UNet', 'num_classes': '5',
                         'num_layers': 2, 'features_start': 32}, 'header': {'_target_': 'torch.nn.Identity'}},
        'optimizer': {'_target_': 'torch.optim.Adam', 'lr': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-08,
                      'weight_decay': 0, 'amsgrad': False}, 'callbacks': {
            'check_backbone_header_compatibility': {
                '_target_': 'src.callbacks.model_callbacks.CheckBackboneHeaderCompatibility'},
            'model_checkpoint': {'_target_': 'src.callbacks.model_callbacks.SaveModelStateDictAndTaskCheckpoint',
                                 'monitor': 'val/crossentropyloss', 'save_top_k': 1, 'save_last': True, 'mode': 'min',
                                 'verbose': False, 'dirpath': 'checkpoints/',
                                 'filename': 'dev-baby-unet-cb55-10',
                                 'backbone_filename': 'backbone',
                                 'header_filename': 'header'},
            'watch_model': {'_target_': 'src.callbacks.wandb_callbacks.WatchModelWithWandb', 'log': 'all',
                            'log_freq': 1}}, 'logger': {
            'wandb': {'_target_': 'pytorch_lightning.loggers.wandb.WandbLogger', 'project': 'unsupervised',
                      'name': 'dev-baby-unet-cb55-10', 'offline': False, 'job_type': 'train', 'group': 'dev-runs',
                      'tags': ['best_model', 'USL'], 'save_dir': '.', 'log_model': False, 'notes': 'Testing'},
            'csv': {'_target_': 'pytorch_lightning.loggers.csv_logs.CSVLogger', 'save_dir': '.', 'name': 'csv/'}},
        'seed': 42, 'train': True, 'test': True,
        'trainer': {'_target_': 'pytorch_lightning.Trainer', 'gpus': -1, 'accelerator': 'ddp',
                    'min_epochs': 1, 'max_epochs': 3, 'weights_summary': 'full', 'precision': 16},
        'datamodule': {
            '_target_': 'src.datamodules.hisDBDataModule.DIVAHisDBDataModule.DIVAHisDBDataModuleCropped',
            'data_dir': '/net/research-hisdoc/datasets/semantic_segmentation/datasets_cropped/CB55-10-segmentation',
            'crop_size': 256, 'num_workers': 4, 'batch_size': 16, 'shuffle': True, 'drop_last': True},
        'save_config': True, 'checkpoint_folder_name': '{epoch}/', 'work_dir': '.',
        'debug': False, 'print_config': True, 'disable_warnings': True})


def test_check_config_everything_good(get_dict):
    check_config(get_dict)
    assert not get_dict['debug']
    assert get_dict['metric'] == {'_target_': 'src.metrics.divahisdb.HisDBIoU',
                                  'num_classes': '5'}
    assert get_dict['train']
    assert get_dict['test']
    assert 'predict' not in get_dict
    assert get_dict['seed'] == 42


def test_check_config_no_seed(get_dict, caplog):
    del get_dict['seed']
    check_config(get_dict)
    assert get_dict['seed'] > -1
    assert f'No seed specified! Seed set to {get_dict["seed"]}'


def test_check_config_fast_dev_run(get_dict, caplog):
    get_dict['trainer']['fast_dev_run'] = True
    check_config(get_dict)
    assert get_dict['trainer']['gpus'] == 0
    assert get_dict['datamodule']['num_workers'] == 0
    assert "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>" in caplog.text


def test_check_config_no_plugins(get_dict):
    del get_dict['plugins']
    assert 'plugins' not in get_dict
    check_config(get_dict)
    assert 'plugins' in get_dict
    assert get_dict['plugins'] == {
        'ddp_plugin': {'_target_': 'pytorch_lightning.plugins.DDPPlugin', 'find_unused_parameters': False}}


def test_check_config_ddp_cpu_and_precision(get_dict, caplog):
    get_dict['trainer']['accelerator'] = 'ddp_cpu'
    get_dict['trainer']['precision'] = 16
    check_config(get_dict)
    assert 'You are using ddp_cpu without precision=16. This can lead to a crash! Use 64 or 32!' in caplog.text


def test__check_if_in_config_good_config(get_dict):
    for cf in REQUIRED_CONFIGS:
        _check_if_in_config(config=get_dict, name=cf)


def test__check_if_in_config_bad_config(get_dict):
    del get_dict['datamodule']
    with pytest.raises(ValueError):
        for cf in REQUIRED_CONFIGS:
            _check_if_in_config(config=get_dict, name=cf)


def test_print_config(get_dict, capsys):
    print_config(config=get_dict, fields=(
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
    ))
    expected_result = \
        "⚙ CONFIG                                                                        \n" \
        "├── trainer                                                                     \n" \
        "│   └── _target_: pytorch_lightning.Trainer                                     \n" \
        "│       gpus: -1                                                                \n" \
        "│       accelerator: ddp                                                        \n" \
        "│       min_epochs: 1                                                           \n" \
        "│       max_epochs: 3                                                           \n" \
        "│       weights_summary: full                                                   \n" \
        "│       precision: 16                                                           \n" \
        "├── task                                                                        \n" \
        "│   └── _target_: src.tasks.semantic_segmentation.semantic_segmentation.Semantic\n" \
        "│       confusion_matrix_log_every_n_epoch: 1                                   \n" \
        "│       confusion_matrix_val: true                                              \n" \
        "│       confusion_matrix_test: true                                             \n" \
        "├── model                                                                       \n" \
        "│   └── backbone:                                                               \n" \
        "│         _target_: pl_bolts.models.vision.UNet                                 \n" \
        "│         num_classes: '5'                                                      \n" \
        "│         num_layers: 2                                                         \n" \
        "│         features_start: 32                                                    \n" \
        "│       header:                                                                 \n" \
        "│         _target_: torch.nn.Identity                                           \n" \
        "├── optimizer                                                                   \n" \
        "│   └── _target_: torch.optim.Adam                                              \n" \
        "│       lr: 0.001                                                               \n" \
        "│       betas:                                                                  \n" \
        "│       - 0.9                                                                   \n" \
        "│       - 0.999                                                                 \n" \
        "│       eps: 1.0e-08                                                            \n" \
        "│       weight_decay: 0                                                         \n" \
        "│       amsgrad: false                                                          \n" \
        "├── datamodule                                                                  \n" \
        "│   └── _target_: src.datamodules.hisDBDataModule.DIVAHisDBDataModule.DIVAHisDBD\n" \
        "│       data_dir: /net/research-hisdoc/datasets/semantic_segmentation/datasets_c\n" \
        "│       crop_size: 256                                                          \n" \
        "│       num_workers: 4                                                          \n" \
        "│       batch_size: 16                                                          \n" \
        "│       shuffle: true                                                           \n" \
        "│       drop_last: true                                                         \n" \
        "├── callbacks                                                                   \n" \
        "│   └── check_backbone_header_compatibility:                                    \n" \
        "│         _target_: src.callbacks.model_callbacks.CheckBackboneHeaderCompatibili\n" \
        "│       model_checkpoint:                                                       \n" \
        "│         _target_: src.callbacks.model_callbacks.SaveModelStateDictAndTaskCheck\n" \
        "│         monitor: val/crossentropyloss                                         \n" \
        "│         save_top_k: 1                                                         \n" \
        "│         save_last: true                                                       \n" \
        "│         mode: min                                                             \n" \
        "│         verbose: false                                                        \n" \
        "│         dirpath: checkpoints/                                                 \n" \
        "│         filename: dev-baby-unet-cb55-10                                       \n" \
        "│         backbone_filename: backbone                                           \n" \
        "│         header_filename: header                                               \n" \
        "│       watch_model:                                                            \n" \
        "│         _target_: src.callbacks.wandb_callbacks.WatchModelWithWandb           \n" \
        "│         log: all                                                              \n" \
        "│         log_freq: 1                                                           \n" \
        "├── loss                                                                        \n" \
        "│   └── _target_: torch.nn.CrossEntropyLoss                                     \n" \
        "├── metric                                                                      \n" \
        "│   └── _target_: src.metrics.divahisdb.HisDBIoU                                \n" \
        "│       num_classes: '5'                                                        \n" \
        "├── logger                                                                      \n" \
        "│   └── wandb:                                                                  \n" \
        "│         _target_: pytorch_lightning.loggers.wandb.WandbLogger                 \n" \
        "│         project: unsupervised                                                 \n" \
        "│         name: dev-baby-unet-cb55-10                                           \n" \
        "│         offline: false                                                        \n" \
        "│         job_type: train                                                       \n" \
        "│         group: dev-runs                                                       \n" \
        "│         tags:                                                                 \n" \
        "│         - best_model                                                          \n" \
        "│         - USL                                                                 \n" \
        "│         save_dir: .                                                           \n" \
        "│         log_model: false                                                      \n" \
        "│         notes: Testing                                                        \n" \
        "│       csv:                                                                    \n" \
        "│         _target_: pytorch_lightning.loggers.csv_logs.CSVLogger                \n" \
        "│         save_dir: .                                                           \n" \
        "│         name: csv/                                                            \n" \
        "├── seed                                                                        \n" \
        "│   └── 42                                                                      \n" \
        "├── train                                                                       \n" \
        "│   └── True                                                                    \n" \
        "├── test                                                                        \n" \
        "│   └── True                                                                    \n" \
        "├── predict                                                                     \n" \
        "│   └── None                                                                    \n" \
        "├── checkpoint_folder_name                                                      \n" \
        "│   └── {epoch}/                                                                \n" \
        "├── debug                                                                       \n" \
        "│   └── False                                                                   \n" \
        "├── disable_warnings                                                            \n" \
        "│   └── True                                                                    \n" \
        "├── plugins                                                                     \n" \
        "│   └── ddp_plugin:                                                             \n" \
        "│         _target_: pytorch_lightning.plugins.DDPPlugin                         \n" \
        "│         find_unused_parameters: false                                         \n" \
        "├── print_config                                                                \n" \
        "│   └── True                                                                    \n" \
        "├── save_config                                                                 \n" \
        "│   └── True                                                                    \n" \
        "└── work_dir                                                                    \n" \
        "    └── .                                                                       \n"
    assert capsys.readouterr().out == expected_result
