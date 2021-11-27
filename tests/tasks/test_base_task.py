import os

import pytest
import torch
import torchmetrics
from omegaconf import OmegaConf
from pl_bolts.models.vision import UNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.trainer.states import TrainerState, RunningStage
from torch.nn import Identity, CrossEntropyLoss
from torchmetrics import Precision

from src.datamodules.DivaHisDB.datamodule_cropped import DivaHisDBDataModuleCropped
from src.models.backbone_header_model import BackboneHeaderModel
from src.tasks.base_task import AbstractTask
from src.tasks.utils.outputs import OutputKeys
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped


@pytest.fixture(autouse=True)
def clear_resolvers():
    OmegaConf.clear_resolvers()
    seed_everything(42)


@pytest.fixture()
def model():
    return UNet(num_classes=4, num_layers=2, features_start=32)


@pytest.fixture()
def data_module_cropped(data_dir_cropped):
    OmegaConf.clear_resolvers()
    datamodules = DivaHisDBDataModuleCropped(data_dir_cropped, data_folder_name='data', gt_folder_name='gt',
                                             num_workers=4)
    return datamodules


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
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    assert task._get_current_metric() == {}


def test__get_current_metric_train(monkeypatch):
    metric = Precision()
    task = AbstractTask(metric_train=metric)
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    state = TrainerState(stage=RunningStage.TRAINING)
    monkeypatch.setattr(trainer, 'state', state)
    assert dict(task._get_current_metric().items()) == {'precision': Precision()}


def test__get_current_metric_test(monkeypatch):
    metric = Precision()
    task = AbstractTask(metric_test=metric)
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    state = TrainerState(stage=RunningStage.TESTING)
    monkeypatch.setattr(trainer, 'state', state)
    assert dict(task._get_current_metric().items()) == {'precision': Precision()}


def test_setup_warning(monkeypatch, data_module_cropped, caplog):
    stage = 'something'
    task = AbstractTask()
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)

    task.setup(stage)

    assert f'Unknown stage ({stage}) during setup!' in caplog.text
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_test(monkeypatch, data_module_cropped):
    stage = 'test'
    task = AbstractTask()
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


@pytest.mark.skip(reason='predict not implemented for DivaHisDBDataModuleCropped')
def test_setup_predict(monkeypatch, data_module_cropped):
    stage = 'predict'
    task = AbstractTask()
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_fit(monkeypatch, data_module_cropped):
    stage = 'fit'
    task = AbstractTask()
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'trainer', trainer)
    data_module_cropped.setup(stage)
    task.setup(stage)
    assert not hasattr(task, 'metric_conf_mat_val')
    assert not hasattr(task, 'metric_conf_mat_test')


def test_setup_conf_mats(monkeypatch, data_module_cropped):
    task = AbstractTask(confusion_matrix_val=True, confusion_matrix_test=True)
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    task.setup(stage='')
    assert task.confusion_matrix_val is not None
    assert isinstance(task.metric_conf_mat_val, torchmetrics.ConfusionMatrix)
    assert task.confusion_matrix_test is not None
    assert isinstance(task.metric_conf_mat_test, torchmetrics.ConfusionMatrix)


def test_step(monkeypatch, data_module_cropped, model):
    # setup
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    data_module_cropped.setup('fit')

    img, gt, _ = data_module_cropped.train[0]
    output = task.step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert output[OutputKeys.LOSS].item() == 1.4348618984222412
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test__create_conf_mat(monkeypatch, data_module_cropped, model, tmp_path):
    # setup
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss(), confusion_matrix_test=True)
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    data_module_cropped.setup('fit')
    task.setup('fit')
    monkeypatch.setattr(trainer, 'val_dataloaders', [data_module_cropped.val_dataloader()])
    os.chdir(str(tmp_path))

    img, gt, _ = data_module_cropped.val[0]
    task.step(batch=(img[None, :], gt[None, :]), batch_idx=0)

    hist = task.metric_conf_mat_test.compute()
    hist = hist.detach().numpy()
    task._create_conf_mat(matrix=hist, stage='val')
    assert (tmp_path / 'conf_mats').exists()
    assert (tmp_path / 'conf_mats' / 'val').exists()
    assert (tmp_path / 'conf_mats' / 'val' / 'CM_epoch_0.txt').exists()


def test_forward(monkeypatch, model, data_module_cropped):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    data_module_cropped.setup('fit')

    img, _, _ = data_module_cropped.train[0]
    out = task(img[None, :])
    assert out.shape == torch.Size([1, 4, 256, 256])
    assert out.ndim == 4


def test_training_step(monkeypatch, model, data_module_cropped, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('fit')

    img, gt, _ = data_module_cropped.train[0]
    output = task.training_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert capsys.readouterr().out == 'train/crossentropyloss 1.4348618984222412\n'
    assert output[OutputKeys.LOSS].item() == 1.4348618984222412
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test_validation_step(monkeypatch, model, data_module_cropped, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('fit')

    img, gt, _ = data_module_cropped.val[0]
    output = task.validation_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert capsys.readouterr().out == 'val/crossentropyloss 1.4348618984222412\n'
    assert output[OutputKeys.LOSS].item() == 1.4348618984222412
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test_test_step(monkeypatch, model, data_module_cropped, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('test')

    img, gt, _, _ = data_module_cropped.test[0]
    output = task.test_step(batch=(img[None, :], gt[None, :]), batch_idx=0)
    assert capsys.readouterr().out == 'test/crossentropyloss 1.428513765335083\n'
    assert output[OutputKeys.LOSS].item() == 1.428513765335083
    assert torch.equal(output[OutputKeys.TARGET], gt[None, :])
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])


def test_predict_step(monkeypatch, model, data_module_cropped, capsys):
    task = AbstractTask(model=BackboneHeaderModel(backbone=model, header=Identity()),
                        loss_fn=CrossEntropyLoss())
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    data_module_cropped.setup('test')

    img, _, _, _ = data_module_cropped.test[0]
    output = task.predict_step(batch=img[None, :], batch_idx=0)
    assert output[OutputKeys.PREDICTION].shape == torch.Size([1, 4, 256, 256])
    assert output[OutputKeys.PREDICTION].ndim == 4


def fake_log(name, value, on_epoch=True, on_step=True, sync_dist=True, rank_zero_only=True):
    print(name, value.item())
