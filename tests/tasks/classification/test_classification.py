import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from torch.nn import CrossEntropyLoss, Identity

from src.datamodules.RotNet.datamodule_cropped import RotNetDivaHisDBDataModuleCropped
from src.models.backbone_header_model import BackboneHeaderModel
from src.models.backbones.baby_cnn import CNN_basic
from src.models.headers.fully_connected import SingleLinear
from src.tasks.classification.classification import Classification
from src.tasks.utils.outputs import OutputKeys
from tasks.test_base_task import fake_log
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


def test_classification(tmp_path, task, datamodule_and_dir):
    data_module, _ = datamodule_and_dir

    trainer = pl.Trainer(max_epochs=2, precision=32, default_root_dir=task.test_output_path,
                         accelerator='ddp_cpu')

    trainer.fit(task, datamodule=data_module)

    results = trainer.test()
    print(results)
    assert np.isclose(results[0]['test/crossentropyloss'], 1.5777363777160645, rtol=2e-03)
    assert np.isclose(results[0]['test/crossentropyloss_epoch'], 1.5777363777160645, rtol=2e-03)


def test_setup_train(monkeypatch, datamodule_and_dir, model):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    stage = 'fit'
    task = Classification(model=model, optimizer=torch.optim.Adam(model.parameters()))
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_test(monkeypatch, datamodule_and_dir, model):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    stage = 'test'
    task = Classification(model=model, optimizer=torch.optim.Adam(model.parameters()))
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_training_step(monkeypatch, datamodule_and_dir, model, capsys):
    task = Classification(model=BackboneHeaderModel(backbone=model, header=Identity()),
                          loss_fn=CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()))
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('fit')

    img, gt = data_module_cropped.train[0]
    gt_tensor = torch.as_tensor([gt])
    output = task.training_step(batch=(img[None, :], gt_tensor), batch_idx=0)
    assert capsys.readouterr().out == 'train/crossentropyloss 1.3173744678497314\n'
    assert output[OutputKeys.LOSS].item() == 1.3173744678497314


def test_validation_step(monkeypatch, datamodule_and_dir, model, capsys):
    task = Classification(model=BackboneHeaderModel(backbone=model, header=Identity()),
                          loss_fn=CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()))
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('fit')

    img, gt = data_module_cropped.val[0]
    gt_tensor = torch.as_tensor([gt])
    output = task.validation_step(batch=(img[None, :], gt_tensor), batch_idx=0)
    assert capsys.readouterr().out == 'val/crossentropyloss 1.3173744678497314\n'
    assert not output


def test_test_step(monkeypatch, datamodule_and_dir, model, capsys):
    task = Classification(model=BackboneHeaderModel(backbone=model, header=Identity()),
                          loss_fn=CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()))
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('test')

    img, gt = data_module_cropped.test[0]
    gt_tensor = torch.as_tensor([gt])
    output = task.test_step(batch=(img[None, :], gt_tensor), batch_idx=0)
    assert capsys.readouterr().out == 'test/crossentropyloss 1.240635633468628\n'
    assert not output
