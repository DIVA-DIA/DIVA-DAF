import os

import numpy as np
import pytest
import torch
import torchmetrics
from omegaconf import OmegaConf
from pl_bolts.models.vision import UNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.trainer.states import TrainerState, RunningStage
from torch.nn import Identity, CrossEntropyLoss
from torchmetrics import Precision, MetricCollection

from src.models.backbone_header_model import BackboneHeaderModel
from src.tasks.base_task import AbstractTask
from src.tasks.utils.outputs import OutputKeys
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped
from tests.datamodules.DivaHisDB.test_hisDBDataModule import data_module_cropped_hisdb


@pytest.fixture(autouse=True)
def clear_resolvers():
    OmegaConf.clear_resolvers()
    seed_everything(42)


@pytest.fixture()
def model():
    return UNet(num_classes=4, num_layers=2, features_start=32)


def test_to_loss_format():
    x = torch.as_tensor([1, 2, 3])
    y = AbstractTask.to_loss_format(x)
    assert torch.equal(x, y)


def test_to_metrics_format():
    x = torch.as_tensor([1, 2, 3])
    y = AbstractTask.to_metrics_format(x)
    assert torch.equal(x, y)


def test__get_current_metric_fail(monkeypatch):
    task = AbstractTask()
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    assert task._get_current_metric() == {}


def test__get_current_metric_train(monkeypatch):
    metric = MetricCollection(Precision())
    task = AbstractTask(metric_train=metric)
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    state = TrainerState(stage=RunningStage.TRAINING)
    monkeypatch.setattr(trainer, 'state', state)
    assert dict(task._get_current_metric().items()) == {'Precision': Precision()}


def test__get_current_metric_test(monkeypatch):
    metric = MetricCollection(Precision())
    task = AbstractTask(metric_test=metric)
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    state = TrainerState(stage=RunningStage.TESTING)
    monkeypatch.setattr(trainer, 'state', state)
    assert dict(task._get_current_metric().items()) == {'Precision': Precision()}


def test_setup_warning(monkeypatch, data_module_cropped_hisdb, caplog):
    stage = 'something'
    task = AbstractTask()
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)

    task.setup(stage)

    assert f'Unknown stage ({stage}) during setup!' in caplog.text
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_test(monkeypatch, data_module_cropped_hisdb):
    stage = 'test'
    task = AbstractTask()
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped_hisdb.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


@pytest.mark.skip(reason='predict not implemented for DivaHisDBDataModuleCropped')
def test_setup_predict(monkeypatch, data_module_cropped_hisdb):
    stage = 'predict'
    task = AbstractTask()
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped_hisdb.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_fit(monkeypatch, data_module_cropped_hisdb):
    stage = 'fit'
    task = AbstractTask()
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped_hisdb.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_conf_mats(monkeypatch, data_module_cropped_hisdb):
    task = AbstractTask(confusion_matrix_val=True, confusion_matrix_test=True)
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    task.setup(stage='')
    assert task.confusion_matrix_val is not None
    assert isinstance(task.metric_conf_mat_val, torchmetrics.ConfusionMatrix)
    assert task.confusion_matrix_test is not None
    assert isinstance(task.metric_conf_mat_test, torchmetrics.ConfusionMatrix)


def test_step(monkeypatch, data_module_cropped_hisdb, model):
    # setup
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    data_module_cropped_hisdb.setup('fit')

    img, gt, _ = data_module_cropped_hisdb.train[0]
    output = task.step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert torch.isclose(output[OutputKeys.LOSS], torch.tensor(1.4348618984222412).type_as(output[OutputKeys.LOSS]))
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test__create_conf_mat_test_error(monkeypatch, data_module_cropped_hisdb, model, tmp_path):
    # setup
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss(), confusion_matrix_test=True)
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    data_module_cropped_hisdb.setup('test')
    task.setup('test')
    monkeypatch.setattr(trainer, 'test_dataloaders', [data_module_cropped_hisdb.test_dataloader()])
    os.chdir(str(tmp_path))

    img, gt, _, _ = data_module_cropped_hisdb.test[0]
    task.step(batch=(img[None, :], gt[None, :]), batch_idx=0)

    hist = task.metric_conf_mat_test.compute()
    hist = hist.detach().numpy()
    with pytest.raises(ValueError):
        task._create_conf_mat(matrix=hist, stage='something')


def test__create_conf_mat_test(monkeypatch, data_module_cropped_hisdb, model, tmp_path):
    # setup
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss(), confusion_matrix_test=True)
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    data_module_cropped_hisdb.setup('test')
    task.setup('test')
    monkeypatch.setattr(trainer, 'test_dataloaders', [data_module_cropped_hisdb.test_dataloader()])
    os.chdir(str(tmp_path))

    img, gt, _, _ = data_module_cropped_hisdb.test[0]
    task.step(batch=(img[None, :], gt[None, :]), batch_idx=0)

    hist = task.metric_conf_mat_test.compute()
    hist = hist.detach().numpy()
    task._create_conf_mat(matrix=hist, stage='test')
    assert (tmp_path / 'conf_mats').exists()
    assert (tmp_path / 'conf_mats' / 'test').exists()
    assert (tmp_path / 'conf_mats' / 'test' / 'CM_epoch_0.txt').exists()


def test__create_conf_mat_val_not_drop_last(monkeypatch, data_module_cropped_hisdb, model, tmp_path):
    # setup
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss(), confusion_matrix_test=True)
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(data_module_cropped_hisdb, 'drop_last', False)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    data_module_cropped_hisdb.setup('fit')
    task.setup('fit')
    monkeypatch.setattr(trainer, 'val_dataloaders', [data_module_cropped_hisdb.val_dataloader()])
    os.chdir(str(tmp_path))

    img, gt, _ = data_module_cropped_hisdb.val[0]
    task.step(batch=(img[None, :], gt[None, :]), batch_idx=0)

    hist = task.metric_conf_mat_test.compute()
    hist = hist.detach().numpy()
    task._create_conf_mat(matrix=hist, stage='val')
    assert (tmp_path / 'conf_mats').exists()
    assert (tmp_path / 'conf_mats' / 'val').exists()
    assert (tmp_path / 'conf_mats' / 'val' / 'CM_epoch_0.txt').exists()


def test__create_conf_mat_val_drop_last(monkeypatch, data_module_cropped_hisdb, model, tmp_path):
    # setup
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss(), confusion_matrix_test=True)
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    data_module_cropped_hisdb.setup('fit')
    task.setup('fit')
    monkeypatch.setattr(trainer, 'val_dataloaders', [data_module_cropped_hisdb.val_dataloader()])
    os.chdir(str(tmp_path))

    img, gt, _ = data_module_cropped_hisdb.val[0]
    task.step(batch=(img[None, :], gt[None, :]), batch_idx=0)

    hist = task.metric_conf_mat_test.compute()
    hist = hist.detach().numpy()
    task._create_conf_mat(matrix=hist, stage='val')
    assert (tmp_path / 'conf_mats').exists()
    assert (tmp_path / 'conf_mats' / 'val').exists()
    assert (tmp_path / 'conf_mats' / 'val' / 'CM_epoch_0.txt').exists()


def test_forward(monkeypatch, model, data_module_cropped_hisdb):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    data_module_cropped_hisdb.setup('fit')

    img, _, _ = data_module_cropped_hisdb.train[0]
    out = task(img[None, :])
    assert out.shape == torch.Size([1, 4, 256, 256])
    assert out.ndim == 4


def test_training_step(monkeypatch, model, data_module_cropped_hisdb, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped_hisdb.setup('fit')

    img, gt, _ = data_module_cropped_hisdb.train[0]
    output = task.training_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert 'train/crossentropyloss 1.4' in capsys.readouterr().out
    assert np.isclose(output[OutputKeys.LOSS].item(), 1.4348618984222412, rtol=2e-03)
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test_validation_step(monkeypatch, model, data_module_cropped_hisdb, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped_hisdb.setup('fit')

    img, gt, _ = data_module_cropped_hisdb.val[0]
    output = task.validation_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert 'val/crossentropyloss 1.4' in capsys.readouterr().out
    assert np.isclose(output[OutputKeys.LOSS].item(), 1.4348618984222412, rtol=2e-03)
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test_test_step(monkeypatch, model, data_module_cropped_hisdb, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped_hisdb.setup('test')

    img, gt, _, _ = data_module_cropped_hisdb.test[0]
    output = task.test_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert 'test/crossentropyloss 1.4' in capsys.readouterr().out
    assert np.isclose(output[OutputKeys.LOSS].item(), 1.428513765335083, rtol=2e-03)
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test_predict_step(monkeypatch, model, data_module_cropped_hisdb, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer(accelerator='cpu', strategy='ddp')
    monkeypatch.setattr(data_module_cropped_hisdb, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_hisdb)
    data_module_cropped_hisdb.setup('test')

    img, _, _, _ = data_module_cropped_hisdb.test[0]
    output = task.predict_step(batch=img[None, :], batch_idx=0)
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])
    assert output[OutputKeys.PREDICTION].ndim == 4


def fake_log(name, value, on_epoch=True, on_step=True, sync_dist=True, rank_zero_only=True):
    print(name, value.item())
