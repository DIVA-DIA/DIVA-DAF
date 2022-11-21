import pytest
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from src.datamodules.IndexedFormats.datamodule import DataModuleIndexed
from tests.test_data.dummy_fixed_gif.dummy_data import data_dir


@pytest.fixture
def datamodule_indexed(data_dir):
    OmegaConf.clear_resolvers()
    datamodules = DataModuleIndexed(data_dir, data_folder_name='data', gt_folder_name='gt',
                                    num_workers=4)
    return datamodules


def test_setup_fit(datamodule_indexed, monkeypatch):
    stage = 'fit'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(datamodule_indexed, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', datamodule_indexed)
    monkeypatch.setattr(datamodule_indexed, 'batch_size', 1)
    datamodule_indexed.setup(stage)
    assert not hasattr(datamodule_indexed, 'test')
    assert hasattr(datamodule_indexed, 'train')
    assert hasattr(datamodule_indexed, 'val')
    assert not hasattr(datamodule_indexed, 'predict')


def test_setup_fit_error(datamodule_indexed, monkeypatch, caplog):
    stage = 'fit'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(datamodule_indexed, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', datamodule_indexed)
    with pytest.raises(ValueError):
        datamodule_indexed.setup(stage)


def test_setup_test(datamodule_indexed, monkeypatch):
    stage = 'test'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(datamodule_indexed, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', datamodule_indexed)
    datamodule_indexed.setup(stage)
    assert hasattr(datamodule_indexed, 'test')
    assert not hasattr(datamodule_indexed, 'train')
    assert not hasattr(datamodule_indexed, 'val')
    assert not hasattr(datamodule_indexed, 'predict')


def test_setup_predict(data_dir, monkeypatch):
    stage = 'predict'
    pred_file_path_list = [str(data_dir / 'test' / 'data' / 'fmb-cb-55-011r.jpg')]
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    datamodule_indexed = DataModuleIndexed(data_dir, data_folder_name='data', gt_folder_name='gt', num_workers=4,
                                           pred_file_path_list=pred_file_path_list)

    monkeypatch.setattr(datamodule_indexed, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', datamodule_indexed)
    datamodule_indexed.setup(stage)
    assert not hasattr(datamodule_indexed, 'test')
    assert not hasattr(datamodule_indexed, 'train')
    assert not hasattr(datamodule_indexed, 'val')
    assert hasattr(datamodule_indexed, 'predict')


def test_setup_predict_error(data_dir, monkeypatch):
    stage = 'predict'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    datamodule_indexed = DataModuleIndexed(data_dir, data_folder_name='data', gt_folder_name='gt',
                                           num_workers=4, pred_file_path_list=['1', '2'])
    monkeypatch.setattr(datamodule_indexed, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', datamodule_indexed)
    with pytest.raises(RuntimeError):
        datamodule_indexed.setup(stage)

