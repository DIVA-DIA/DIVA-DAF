import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pl_bolts.models.vision import UNet
from pytorch_lightning import seed_everything, Trainer

from tests.datamodules.RolfFormat.datasets.test_full_page_dataset import _get_dataspecs
from src.datamodules.RolfFormat.datamodule import DataModuleRolfFormat
from src.tasks.RGB.semantic_segmentation import SemanticSegmentationRGB
from src.tasks.utils.outputs import OutputKeys
from tests.tasks.test_base_task import fake_log
from tests.test_data.dummy_data_rolf.dummy_data import data_dir


@pytest.fixture(autouse=True)
def clear_resolvers():
    OmegaConf.clear_resolvers()
    seed_everything(42)
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


@pytest.fixture()
def model():
    return UNet(num_classes=6, num_layers=2, features_start=32)


@pytest.fixture()
def datamodule_and_dir(data_dir):
    # datamodule
    specs_train = _get_dataspecs(data_root=data_dir, train=True).__dict__
    del specs_train['data_root']
    specs_test = _get_dataspecs(data_root=data_dir, train=False).__dict__
    del specs_test['data_root']
    OmegaConf.clear_resolvers()
    pred_file_path_list = [str(data_dir / 'codex' / 'D1-LC-Car-folio-1001.jpg')]
    datamodule = DataModuleRolfFormat(data_dir, train_specs={'a': specs_train}, test_specs={'a': specs_test},
                                      val_specs={'a': specs_train}, num_workers=4, drop_last=False, shuffle=True,
                                      pred_file_path_list=pred_file_path_list)
    return datamodule, data_dir


@pytest.fixture()
def task(model, tmp_path):
    task = SemanticSegmentationRGB(model=model,
                                   optimizer=torch.optim.Adam(params=model.parameters()),
                                   loss_fn=torch.nn.CrossEntropyLoss(),
                                   test_output_path=tmp_path,
                                   confusion_matrix_val=True
                                   )
    return task


def test_semantic_segmentation(tmp_path, task, datamodule_and_dir, monkeypatch):
    data_module, data_dir = datamodule_and_dir
    monkeypatch.chdir(data_dir)

    # different paths needed later
    patches_path = task.test_output_path / 'patches'
    test_data_patch = data_dir / 'test' / 'data'

    trainer = pl.Trainer(max_epochs=2, precision=32, default_root_dir=task.test_output_path,
                         accelerator='cpu', strategy='ddp')

    trainer.fit(task, datamodule=data_module)

    results = trainer.test(datamodule=data_module)
    output_key = "test/crossentropyloss"
    if output_key not in results[0]:
        output_key = "test/crossentropyloss_epoch"
    assert np.isclose(results[0][output_key], 1.861278772354126, rtol=2.5e-02)
    assert len(list(patches_path.glob('*/*.npy'))) == len(list(test_data_patch.glob('*/*.png')))


def test_to_metrics_format():
    x = torch.as_tensor([[1, 2, 3], [0, 1, 2]])
    y = SemanticSegmentationRGB.to_metrics_format(x)
    assert torch.equal(torch.as_tensor([2, 2]), y)


def test_training_step(monkeypatch, datamodule_and_dir, task, capsys):
    data_module, data_dir = datamodule_and_dir
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module.setup('fit')

    img, gt = data_module.train[0]
    output = task.training_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert 'train/crossentropyloss 1.8' in capsys.readouterr().out
    assert np.isclose(output[OutputKeys.LOSS].item(), 1.8575176000595093, rtol=2e-03)


def test_validation_step(monkeypatch, datamodule_and_dir, task, capsys):
    data_module, data_dir = datamodule_and_dir
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module)
    monkeypatch.setattr(task, 'log', fake_log)
    monkeypatch.setattr(task, 'confusion_matrix_val', False)
    data_module.setup('fit')

    img, gt = data_module.val[0]
    task.validation_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert 'val/crossentropyloss 1.8' in capsys.readouterr().out


def test_test_step(monkeypatch, datamodule_and_dir, task, capsys, tmp_path):
    data_module, data_dir = datamodule_and_dir
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module)
    monkeypatch.setattr(task, 'log', fake_log)
    monkeypatch.setattr(task, 'confusion_matrix_val', False)
    monkeypatch.setattr(task, 'test_output_path', tmp_path)
    data_module.setup('test')

    img, gt, idx = data_module.test[0]
    idx_tensor = torch.as_tensor([idx])
    task.test_step(batch=(img[None, :], gt[None, :], idx_tensor), batch_idx=0)
    assert 'test/crossentropyloss 1.8' in capsys.readouterr().out
    assert (tmp_path / 'pred').exists()
    assert (tmp_path / 'pred' / 'D1-LC-Car-folio-1000.gif').exists()
    assert len(list((tmp_path / 'pred').iterdir())) == 1
    assert (tmp_path / 'pred_raw').exists()
    assert (tmp_path / 'pred_raw' / 'D1-LC-Car-folio-1000.npy').exists()
    assert len(list((tmp_path / 'pred_raw').iterdir())) == 1


def test_predict_step(monkeypatch, datamodule_and_dir, task, capsys, tmp_path):
    data_module, data_dir = datamodule_and_dir
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(task, 'predict_output_path', data_dir)
    monkeypatch.setattr(trainer, 'datamodule', data_module)
    monkeypatch.setattr(task, 'log', fake_log)
    monkeypatch.setattr(task, 'confusion_matrix_val', False)
    monkeypatch.setattr(task, 'test_output_path', tmp_path)
    data_module.setup('predict')

    img, idx = data_module.predict[0]
    idx_tensor = torch.as_tensor([idx])
    task.predict_step(batch=(img[None, :], idx_tensor), batch_idx=0)
    assert (tmp_path / 'pred').exists()
    assert (tmp_path / 'pred' / 'D1-LC-Car-folio-1001.gif').exists()
    assert len(list((tmp_path / 'pred').iterdir())) == 1
    assert (tmp_path / 'pred_raw').exists()
    assert (tmp_path / 'pred_raw' / 'D1-LC-Car-folio-1001.npy').exists()
    assert len(list((tmp_path / 'pred_raw').iterdir())) == 1
