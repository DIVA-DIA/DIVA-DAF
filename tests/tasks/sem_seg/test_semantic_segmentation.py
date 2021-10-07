import os

import numpy as np
import pytorch_lightning as pl
import torch.optim.optimizer
from omegaconf import OmegaConf
from pl_bolts.models.vision import UNet
from pytorch_lightning import seed_everything

from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCropped
from src.tasks.semantic_segmentation.semantic_segmentation import SemanticSegmentation

from tests.datamodules.hisDBDataModule.dummy_data.dummy_data import data_dir_cropped


def test_semantic_segmentation(data_dir_cropped, tmp_path):
    OmegaConf.clear_resolvers()
    seed_everything(42)

    # datamodule
    data_module = DIVAHisDBDataModuleCropped(
        data_dir=str(data_dir_cropped),
        batch_size=2, num_workers=2)

    def baby_unet():
        return UNet(num_classes=len(data_module.class_encodings), num_layers=2, features_start=32)

    model = baby_unet()
    segmentation = SemanticSegmentation(model=model,
                                        optimizer=torch.optim.Adam(params=model.parameters()),
                                        loss_fn=torch.nn.CrossEntropyLoss(),
                                        test_output_path=tmp_path
                                        )

    # different paths needed later
    patches_path = segmentation.test_output_path / 'patches'
    test_data_patch = data_dir_cropped / 'test' / 'data'

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    trainer = pl.Trainer(max_epochs=2, precision=32, default_root_dir=segmentation.test_output_path, accelerator='ddp_cpu')

    trainer.fit(segmentation, datamodule=data_module)

    results = trainer.test()
    print(results)
    assert np.isclose(results[0]['test/crossentropyloss'], 1.0625288486480713)
    assert np.isclose(results[0]['test/crossentropyloss_epoch'], 1.0625288486480713)
    assert len(list(patches_path.glob('*/*.npy'))) == len(list(test_data_patch.glob('*/*.png')))
