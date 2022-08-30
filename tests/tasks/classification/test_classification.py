import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer

from src.datamodules.RotNet.datamodule_cropped import RotNetDivaHisDBDataModuleCropped
from src.models.backbones.baby_cnn import CNN_basic
from src.models.headers.fully_connected import SingleLinear
from src.tasks.classification.classification import Classification
from src.tasks.utils.outputs import OutputKeys
from tests.tasks.test_base_task import fake_log
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped


@pytest.fixture(autouse=True)
def clear_resolvers():
    OmegaConf.clear_resolvers()
    seed_everything(42)
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


@pytest.fixture()
def model():
    return torch.nn.Sequential(CNN_basic(), SingleLinear())


@pytest.fixture()
def datamodule_and_dir(data_dir_cropped):
    # datamodule
    data_module = RotNetDivaHisDBDataModuleCropped(
        data_dir=str(data_dir_cropped),
        data_folder_name='data',
        batch_size=2, num_workers=2)
    return data_module, data_dir_cropped


@pytest.fixture()
def task(model, tmp_path):
    task = Classification(model=model,
                          optimizer=torch.optim.Adam(params=model.parameters()),
                          loss_fn=torch.nn.CrossEntropyLoss(),
                          confusion_matrix_val=True
                          )
    return task


def test_classification(tmp_path, task, datamodule_and_dir, monkeypatch):
    data_module, data_dir = datamodule_and_dir
    monkeypatch.chdir(data_dir)

    trainer = pl.Trainer(max_epochs=2, precision=32, default_root_dir=task.test_output_path,
                         accelerator='cpu', strategy='ddp')

    trainer.fit(task, datamodule=data_module)

    results = trainer.test(datamodule=data_module)
    print(results)
    assert np.isclose(results[0]['test/crossentropyloss'], 0.7748091220855713, rtol=2e-02)


def test_setup_train(monkeypatch, datamodule_and_dir, task):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    stage = 'fit'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped.setup(stage)
    task.setup(stage)
    assert hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_test(monkeypatch, datamodule_and_dir, task):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    stage = 'test'
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped.setup(stage)
    task.setup(stage)
    assert hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_training_step(monkeypatch, datamodule_and_dir, task, capsys):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('fit')

    img, gt = data_module_cropped.train[0]
    gt_tensor = torch.as_tensor([gt])
    output = task.training_step(batch=(img[None, :], gt_tensor), batch_idx=0)
    assert 'train/crossentropyloss 1.3' in capsys.readouterr().out
    assert np.isclose(output[OutputKeys.LOSS].item(), 1.3173744678497314, rtol=2e-03)


def test_validation_step(monkeypatch, datamodule_and_dir, task, capsys):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    monkeypatch.setattr(task, 'confusion_matrix_val', False)
    data_module_cropped.setup('fit')

    img, gt = data_module_cropped.val[0]
    gt_tensor = torch.as_tensor([gt])
    output = task.validation_step(batch=(img[None, :], gt_tensor), batch_idx=0)
    assert 'val/crossentropyloss 1.3' in capsys.readouterr().out
    assert not output


def test_test_step(monkeypatch, datamodule_and_dir, task, capsys):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('test')

    img, gt = data_module_cropped.test[0]
    gt_tensor = torch.as_tensor([gt])
    output = task.test_step(batch=(img[None, :], gt_tensor), batch_idx=0)
    assert 'test/crossentropyloss 1.2' in capsys.readouterr().out
    assert not output
