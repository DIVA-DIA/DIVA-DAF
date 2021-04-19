import numpy as np
import pytorch_lightning as pl
from hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCB55
from pl_bolts.models import UNet
from pytorch_lightning import seed_everything

from src.semantic_segmentation import SemanticSegmentation
from tests.dummy_dataset.dummy_data import data_dir


def test_semantic_segmentation(data_dir):
    seed_everything(42)

    # datamodule
    data_module = DIVAHisDBDataModuleCB55(
        data_dir=str(data_dir),
        crops_per_image=10,
        batch_size=2, num_workers=1)

    data_module.setup('test')

    def baby_unet():
        return UNet(num_classes=len(data_module.class_encodings), num_layers=2, features_start=32)

    segmentation = SemanticSegmentation(baby_unet(),
                                        class_encodings=data_module.class_encodings,
                                        img_names_sizes_testset=data_module.his_test.img_names_sizes,
                                        output_path=data_dir / 'baby_unet_testing',
                                        create_confusion_matrix=True)

    # analysis path
    analysis_path = segmentation.output_path / 'analysis'

    # checkpoint requirements
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val-loss',
                                                       dirpath=segmentation.output_path)

    trainer = pl.Trainer(max_epochs=2, log_every_n_steps=10, callbacks=[checkpoint_callback],
                         default_root_dir=segmentation.output_path)

    trainer.fit(segmentation, datamodule=data_module)

    results = trainer.test(datamodule=data_module)
    assert results[0]['test-hisdb-iou'] > 28
    assert results[0]['test-hisdb-iou'] < 32
    assert analysis_path.exists()
    assert len(list(analysis_path.glob('*.png'))) == 5
