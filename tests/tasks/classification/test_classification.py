import os

import numpy as np
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from src.datamodules.RotNet.datamodule_cropped import RotNetDivaHisDBDataModuleCropped

from src.models.backbones.baby_cnn import CNN_basic
from src.tasks.classification.classification import Classification
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped


def test_classification(data_dir_cropped):
    OmegaConf.clear_resolvers()
    seed_everything(42)

    # datamodule
    data_module = RotNetDivaHisDBDataModuleCropped(
        data_dir=str(data_dir_cropped),
        batch_size=2, num_workers=2)

    model = CNN_basic(num_classes=data_module.num_classes)
    task = Classification(model=model,
                          optimizer=torch.optim.Adam(params=model.parameters()),
                          loss_fn=torch.nn.CrossEntropyLoss())

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    trainer = pl.Trainer(max_epochs=2, precision=32, default_root_dir=task.test_output_path, accelerator='ddp_cpu',
                         num_processes=1)

    trainer.fit(task, datamodule=data_module)

    results = trainer.test()
    print(results)
    assert np.isclose(results[0]['test/crossentropyloss'], 1.5777363777160645, rtol=2e-03)
    assert np.isclose(results[0]['test/crossentropyloss_epoch'], 1.5777363777160645, rtol=2e-03)
