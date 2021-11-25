import pytest
from omegaconf import DictConfig

from src.utils.utils import _check_if_in_config, REQUIRED_CONFIGS, check_config


@pytest.fixture
def get_dict():
    return DictConfig({'plugins': {
        'ddp_plugin': {'_target_': 'pytorch_lightning.plugins.DDPPlugin', 'find_unused_parameters': False}}, 'task': {
        '_target_': 'src.tasks.semantic_segmentation.semantic_segmentation.SemanticSegmentation',
        'confusion_matrix_log_every_n_epoch': 1, 'confusion_matrix_val': True, 'confusion_matrix_test': True},
        'loss': {'_target_': 'torch.nn.CrossEntropyLoss'},
        'metric': {'_target_': 'src.metrics.divahisdb.HisDBIoU',
                   'num_classes': '${datamodule:num_classes}'}, 'model': {
            'backbone': {'_target_': 'pl_bolts.models.vision.UNet', 'num_classes': '${datamodule:num_classes}',
                         'num_layers': 2, 'features_start': 32}, 'header': {'_target_': 'torch.nn.Identity'}},
        'optimizer': {'_target_': 'torch.optim.Adam', 'lr': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-08,
                      'weight_decay': 0, 'amsgrad': False}, 'callbacks': {
            'check_backbone_header_compatibility': {
                '_target_': 'src.callbacks.model_callbacks.CheckBackboneHeaderCompatibility'},
            'model_checkpoint': {'_target_': 'src.callbacks.model_callbacks.SaveModelStateDictAndTaskCheckpoint',
                                 'monitor': 'val/crossentropyloss', 'save_top_k': 1, 'save_last': True, 'mode': 'min',
                                 'verbose': False, 'dirpath': 'checkpoints/',
                                 'filename': '${checkpoint_folder_name}dev-baby-unet-cb55-10',
                                 'backbone_filename': '${checkpoint_folder_name}backbone',
                                 'header_filename': '${checkpoint_folder_name}header'},
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
        'save_config': True, 'checkpoint_folder_name': '{epoch}/', 'work_dir': '${hydra:runtime.cwd}',
        'debug': False, 'print_config': True, 'disable_warnings': True})


def test_check_config_everything_good(get_dict):
    check_config(get_dict)


def test_check_config_no_seed(get_dict):
    del get_dict['seed']
    check_config(get_dict)


def test_check_config_no_plugins(get_dict):
    del get_dict['plugins']
    check_config(get_dict)


def test__check_if_in_config_good_config(get_dict):
    for cf in REQUIRED_CONFIGS:
        _check_if_in_config(config=get_dict, name=cf)


def test__check_if_in_config_bad_config(get_dict):
    del get_dict['datamodule']
    with pytest.raises(ValueError):
        for cf in REQUIRED_CONFIGS:
            _check_if_in_config(config=get_dict, name=cf)
