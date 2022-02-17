import pytest
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from tests.datamodules.RolfFormat.datasets.test_full_page_dataset import _get_dataspecs
from src.datamodules.RolfFormat.datamodule import DataModuleRolfFormat
from tests.test_data.dummy_data_rolf.dummy_data import data_dir

NUM_WORKERS = 4


@pytest.fixture
def data_module_rolf(data_dir):
    specs_train = _get_dataspecs(data_root=data_dir, train=True).__dict__
    del specs_train['data_root']
    specs_test = _get_dataspecs(data_root=data_dir, train=False).__dict__
    del specs_test['data_root']
    OmegaConf.clear_resolvers()
    datamodules = DataModuleRolfFormat(data_dir, train_specs={'a': specs_train}, test_specs={'a': specs_test},
                                       val_specs={'a': specs_train}, num_workers=NUM_WORKERS)
    return datamodules


def test__print_image_dims(data_dir, caplog):
    specs = _get_dataspecs(data_root=data_dir, train=True).__dict__
    del specs['data_root']
    dm = DataModuleRolfFormat(data_root=data_dir, train_specs={'a': specs})
    indent = 4 * ' '
    lines = [f'image_dims:', f'{indent}width:  {dm.image_dims.width}', f'{indent}height: {dm.image_dims.height}']

    assert '\n'.join(lines) in caplog.text


def test_setup_fit(data_module_rolf, monkeypatch, caplog):
    stage = 'fit'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_rolf, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rolf)
    monkeypatch.setattr(data_module_rolf, 'batch_size', 1)
    data_module_rolf.setup(stage)
    assert not hasattr(data_module_rolf, 'test')
    assert hasattr(data_module_rolf, 'train')
    assert hasattr(data_module_rolf, 'val')
    assert not hasattr(data_module_rolf, 'predict')


def test_setup_fit_error(data_module_rolf, monkeypatch, caplog):
    stage = 'fit'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_rolf, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rolf)
    with pytest.raises(ValueError):
        data_module_rolf.setup(stage)


def test_setup_test(data_module_rolf, monkeypatch):
    stage = 'test'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_rolf, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_rolf)
    data_module_rolf.setup(stage)
    assert hasattr(data_module_rolf, 'test')
    assert not hasattr(data_module_rolf, 'train')
    assert not hasattr(data_module_rolf, 'val')
    assert not hasattr(data_module_rolf, 'predict')


def test_setup_predict(data_dir, monkeypatch):
    stage = 'predict'
    pred_file_path_list = [str(data_dir / 'codex' / 'D1-LC-Car-folio-1001.jpg')]
    specs_train = _get_dataspecs(data_root=data_dir, train=True).__dict__
    del specs_train['data_root']
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    datamodules = DataModuleRolfFormat(data_dir, train_specs={'a': specs_train},
                                       pred_file_path_list=pred_file_path_list,
                                       num_workers=NUM_WORKERS)
    monkeypatch.setattr(datamodules, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', datamodules)
    datamodules.setup(stage)
    assert not hasattr(datamodules, 'test')
    assert not hasattr(datamodules, 'train')
    assert not hasattr(datamodules, 'val')
    assert hasattr(datamodules, 'predict')


def test_setup_predict_error(data_dir, monkeypatch):
    stage = 'predict'
    pred_file_path_list = ['1', '2']
    specs_train = _get_dataspecs(data_root=data_dir, train=True).__dict__
    del specs_train['data_root']
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    datamodules = DataModuleRolfFormat(data_dir, train_specs={'a': specs_train},
                                       pred_file_path_list=pred_file_path_list,
                                       num_workers=NUM_WORKERS)
    monkeypatch.setattr(datamodules, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', datamodules)
    with pytest.raises(RuntimeError):
        datamodules.setup(stage)
