import os

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from src.datamodules.RotNet.datamodule_cropped import RotNetDivaHisDBDataModuleCropped
from src.models.backbones.baby_cnn import CNN_basic
from src.models.headers.fully_connected import SingleLinear
from src.tasks.classification.classification import Classification
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
