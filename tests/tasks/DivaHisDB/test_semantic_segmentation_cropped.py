import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pl_bolts.models.vision import UNet
from pytorch_lightning import seed_everything, Trainer

from src.datamodules.DivaHisDB.datamodule_cropped import DivaHisDBDataModuleCropped
from src.tasks.DivaHisDB.semantic_segmentation_cropped import SemanticSegmentationCroppedHisDB
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
    return UNet(num_classes=4, num_layers=2, features_start=32)


@pytest.fixture()
def datamodule_and_dir(data_dir_cropped):
    # datamodule
    data_module = DivaHisDBDataModuleCropped(
        data_dir=str(data_dir_cropped),
        data_folder_name='data', gt_folder_name='gt',
        batch_size=2, num_workers=2)
    return data_module, data_dir_cropped


@pytest.fixture()
def task(model, tmp_path):
    task = SemanticSegmentationCroppedHisDB(model=model,
                                            optimizer=torch.optim.Adam(params=model.parameters()),
                                            loss_fn=torch.nn.CrossEntropyLoss(),
                                            test_output_path=tmp_path,
                                            confusion_matrix_val=True
                                            )
    return task


def test_semantic_segmentation(tmp_path, task, datamodule_and_dir, monkeypatch):
    data_module, data_dir_cropped = datamodule_and_dir
    monkeypatch.chdir(data_dir_cropped)

    # different paths needed later
    patches_path = task.test_output_path / 'patches'
    test_data_patch = data_dir_cropped / 'test' / 'data'

    trainer = pl.Trainer(max_epochs=2, precision=32, default_root_dir=task.test_output_path,
                         accelerator='cpu', strategy='ddp')

    trainer.fit(task, datamodule=data_module)

    results = trainer.test(datamodule=data_module)
    print(results)
    assert np.isclose(results[0]['test/crossentropyloss'], 1.0896027088165283, rtol=2.5e-02)
    assert len(list(patches_path.glob('*/*.npy'))) == len(list(test_data_patch.glob('*/*.png')))


def test_to_metrics_format():
    x = torch.as_tensor([[1, 2, 3], [0, 1, 2]])
    y = SemanticSegmentationCroppedHisDB.to_metrics_format(x)
    assert torch.equal(torch.as_tensor([2, 2]), y)


def test_training_step(monkeypatch, datamodule_and_dir, task, capsys):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    data_module_cropped.setup('fit')

    img, gt, mask = data_module_cropped.train[0]
    output = task.training_step(batch=(img[None, :], gt[None, :], mask[None, :]), batch_idx=0)
    assert 'train/crossentropyloss 1.4' in capsys.readouterr().out
    assert np.isclose(output[OutputKeys.LOSS].item(), 1.4348618984222412, rtol=2e-03)


def test_validation_step(monkeypatch, datamodule_and_dir, task, capsys):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    monkeypatch.setattr(task, 'confusion_matrix_val', False)
    data_module_cropped.setup('fit')

    img, gt, mask = data_module_cropped.val[0]
    task.validation_step(batch=(img[None, :], gt[None, :], mask[None, :]), batch_idx=0)
    assert 'val/crossentropyloss 1.4' in capsys.readouterr().out


def test_test_step(monkeypatch, datamodule_and_dir, task, capsys, tmp_path):
    data_module_cropped, data_dir_cropped = datamodule_and_dir
    trainer = Trainer()
    monkeypatch.setattr(data_module_cropped, 'trainer', trainer)
    monkeypatch.setattr(task, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped)
    monkeypatch.setattr(task, 'log', fake_log)
    monkeypatch.setattr(task, 'confusion_matrix_val', False)
    monkeypatch.setattr(task, 'test_output_path', tmp_path)
    data_module_cropped.setup('test')

    img, gt, mask, idx = data_module_cropped.test[0]
    idx_tensor = torch.as_tensor([idx])
    task.test_step(batch=(img[None, :], gt[None, :], mask[None, :], idx_tensor), batch_idx=0)
    assert 'test/crossentropyloss 1.4' in capsys.readouterr().out
    assert (tmp_path / 'patches').exists()
    assert (tmp_path / 'patches' / 'e-codices_fmb-cb-0055_0098v_max').exists()
    assert len(list((tmp_path / 'patches' / 'e-codices_fmb-cb-0055_0098v_max').iterdir())) == 1
