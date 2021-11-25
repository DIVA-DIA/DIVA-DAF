import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pl_bolts.models.vision import UNet
from pytorch_lightning import seed_everything

from src.datamodules.DivaHisDB.datamodule_cropped import DivaHisDBDataModuleCropped
from src.tasks.DivaHisDB.semantic_segmentation_cropped import SemanticSegmentationCroppedHisDB
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


def test_semantic_segmentation(tmp_path, task, datamodule_and_dir):
    data_module, data_dir_cropped = datamodule_and_dir

    # different paths needed later
    patches_path = task.test_output_path / 'patches'
    test_data_patch = data_dir_cropped / 'test' / 'data'

    trainer = pl.Trainer(max_epochs=2, precision=32, default_root_dir=task.test_output_path,
                         accelerator='ddp_cpu')

    trainer.fit(task, datamodule=data_module)

    results = trainer.test()
    print(results)
    assert np.isclose(results[0]['test/crossentropyloss'], 1.0896027088165283, rtol=2e-03)
    assert np.isclose(results[0]['test/crossentropyloss_epoch'], 1.0896027088165283, rtol=2e-03)
    assert len(list(patches_path.glob('*/*.npy'))) == len(list(test_data_patch.glob('*/*.png')))
