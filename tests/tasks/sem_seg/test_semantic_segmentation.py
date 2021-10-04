import pytorch_lightning as pl
import torch.optim.optimizer
from pl_bolts.models.vision import UNet
from pytorch_lightning import seed_everything

from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCropped
from src.tasks.semantic_segmentation.semantic_segmentation import SemanticSegmentation

from tests.datamodules.hisDBDataModule.dummy_data.dummy_data import data_dir_cropped


def test_semantic_segmentation(data_dir_cropped):
    seed_everything(42)

    # datamodule
    data_module = DIVAHisDBDataModuleCropped(
        data_dir=str(data_dir_cropped),
        batch_size=2, num_workers=2)

    def baby_unet():
        return UNet(num_classes=len(data_module.class_encodings), num_layers=2, features_start=32)

    model = baby_unet()
    segmentation = SemanticSegmentation(model=model,
                                        optimizer=torch.optim.Adam(params=model.parameters(),
                                                                   lr=1e-3,
                                                                   betas=[0.9, 0.999],
                                                                   eps=1e-8,
                                                                   weight_decay=0,
                                                                   amsgrad=False),
                                        loss_fn=torch.nn.CrossEntropyLoss()
                                        )

    # different paths needed later
    analysis_path = segmentation.test_output_path / 'analysis'
    patches_path = segmentation.test_output_path / 'patches'
    test_data_patch = data_dir_cropped / 'test' / 'data'

    trainer = pl.Trainer(max_epochs=2, log_every_n_steps=10,
                         default_root_dir=segmentation.test_output_path, accelerator='ddp_cpu')

    # TODO fix numbers again
    assert 1 == trainer.fit(segmentation, datamodule=data_module)

    results = trainer.test()
    assert results[0]['test-hisdb-iou'] > 38
    assert results[0]['test-hisdb-iou'] < 41
    assert analysis_path.exists()
    assert len(list(analysis_path.glob('*.png'))) == 5
    assert len(list(patches_path.glob('*/*.npy'))) == len(list(test_data_patch.glob('*/*.png')))
