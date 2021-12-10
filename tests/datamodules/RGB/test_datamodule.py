import pytest
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from src.datamodules.RGB.datamodule import DataModuleRGB
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir

NUM_WORKERS = 4


@pytest.fixture
def data_module_rgb(data_dir):
    OmegaConf.clear_resolvers()
    datamodules = DataModuleRGB(data_dir, data_folder_name='data', gt_folder_name='gt',
                                num_workers=NUM_WORKERS)
    return datamodules


def test_setup_fit(data_module_rgb, monkeypatch, caplog):
    stage = 'fit'
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_rgb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rgb)
    monkeypatch.setattr(data_module_rgb, 'batch_size', 1)
    data_module_rgb.setup(stage)
    assert not hasattr(data_module_rgb, 'test')
    assert hasattr(data_module_rgb, 'train')
    assert hasattr(data_module_rgb, 'val')
    assert not hasattr(data_module_rgb, 'predict')


def test_setup_fit_error(data_module_rgb, monkeypatch, caplog):
    stage = 'fit'
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_rgb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rgb)
    with pytest.raises(ValueError):
        data_module_rgb.setup(stage)


def test_setup_test(data_module_rgb, monkeypatch):
    stage = 'test'
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_rgb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rgb)
    data_module_rgb.setup(stage)
    assert hasattr(data_module_rgb, 'test')
    assert not hasattr(data_module_rgb, 'train')
    assert not hasattr(data_module_rgb, 'val')
    assert not hasattr(data_module_rgb, 'predict')


def test_setup_predict(data_dir, monkeypatch):
    stage = 'predict'
    pred_file_path_list = [str(data_dir / 'test' / 'data' / 'e-codices_fmb-cb-0055_0098v_max.jpg')]
    trainer = Trainer(accelerator='ddp')
    data_module_rgb = DataModuleRGB(data_dir, data_folder_name='data', gt_folder_name='gt', num_workers=NUM_WORKERS,
                                    pred_file_path_list=pred_file_path_list)

    monkeypatch.setattr(data_module_rgb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rgb)
    data_module_rgb.setup(stage)
    assert not hasattr(data_module_rgb, 'test')
    assert not hasattr(data_module_rgb, 'train')
    assert not hasattr(data_module_rgb, 'val')
    assert hasattr(data_module_rgb, 'predict')


def test_setup_predict_error(data_dir, monkeypatch):
    stage = 'predict'
    trainer = Trainer(accelerator='ddp')
    data_module_rgb = DataModuleRGB(data_dir, data_folder_name='data', gt_folder_name='gt',
                                    num_workers=NUM_WORKERS, pred_file_path_list=['1', '2'])
    monkeypatch.setattr(data_module_rgb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rgb)
    with pytest.raises(RuntimeError):
        data_module_rgb.setup(stage)
