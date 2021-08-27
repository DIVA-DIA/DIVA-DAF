from pathlib import Path

import pytorch_lightning as pl
from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCropped
from pl_bolts.models.vision import UNet
from pytorch_lightning import seed_everything

from src.models.semantic_segmentation.semantic_segmentation import SemanticSegmentation

if __name__ == '__main__':
    # because of ddp
    seed_everything(42)

    run_big_test = False

    if run_big_test:
        # BIG TEST
        # datamodule
        data_module = DIVAHisDBDataModuleCropped(data_dir='/dataset/DIVA-HisDB/segmentation/CB55',
                                                 batch_size=16, num_workers=4)

        # load the img_names_sizes information as suggested here
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#using-a-datamodule
        data_module.setup('test')

        model = SemanticSegmentation.load_from_checkpoint(
            '/data/usl_experiments/semantic_segmentation/supervised/unet_crops_1000_CB55/2021-01-05_19-01-24/epoch=39-step=102399.ckpt',
            class_encodings=data_module.class_encodings, model=UNet(num_classes=len(data_module.class_encodings)),
            img_names_sizes_testset=data_module.his_test.img_names_sizes)

        model.test_output_path = Path('/data/usl_experiments/tmp_testing_output/unet_crops_1000_CB55/')

        trainer = pl.Trainer(gpus=-1, accelerator='ddp', max_epochs=50, log_every_n_steps=10)
        trainer.test(model=model, datamodule=data_module)

    else:
        # SMALL TEST
        # datamodule
        data_module = DIVAHisDBDataModuleCropped(
            data_dir='/data/usl_experiments/semantic_segmentation/datasets_cropped/CB55-10-segmentation',
            batch_size=16, num_workers=4)

        def baby_unet():
            return UNet(num_classes=len(data_module.class_encodings), num_layers=2, features_start=32)

        # load the img_names_sizes information as suggested here
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#using-a-datamodule
        data_module.setup('test')

        model = SemanticSegmentation.load_from_checkpoint(
            '/data/usl_experiments/tmp_testing_output/baby_unet_cropped_cb55_v2021_04_22a/epoch=4-step=19.ckpt',
            class_encodings=data_module.class_encodings,
            model=baby_unet())

        model.test_output_path = Path('/data/usl_experiments/tmp_testing_output/baby_unet_cropped_cb55_v2021_04_22a/')

        trainer = pl.Trainer(gpus=-1, accelerator='ddp', max_epochs=50, log_every_n_steps=10)
        trainer.test(model=model, datamodule=data_module)
