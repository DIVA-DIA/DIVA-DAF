from pathlib import Path

import pytorch_lightning as pl
from hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCB55
from pl_bolts.models import UNet
from pytorch_lightning import seed_everything

from src.semantic_segmentation import SemanticSegmentation

if __name__ == '__main__':
    # because of ddp
    seed_everything(42)

    def baby_unet():
        return UNet(num_classes=len(data_module.class_encodings), num_layers=2, features_start=32)

    run_big_test = True

    if run_big_test:
        # BIG TEST
        # datamodule
        data_module = DIVAHisDBDataModuleCB55(data_dir='/dataset/DIVA-HisDB/segmentation/CB55',
                                              crops_per_image=100, num_workers=4, batch_size=8)

        # load the img_names_sizes information as suggested here
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#using-a-datamodule
        data_module.setup('test')

        model = SemanticSegmentation.load_from_checkpoint(
            '/data/usl_experiments/semantic_segmentation/supervised/unet_crops_1000_CB55/2021-01-05_19-01-24/epoch=39-step=102399.ckpt',
            class_encodings=data_module.class_encodings, model=UNet(num_classes=len(data_module.class_encodings)),
            img_names_sizes_testset=data_module.his_test.img_names_sizes)

        model.output_path = Path('/data/usl_experiments/tmp_testing_output/unet_crops_1000_CB55/')

        trainer = pl.Trainer(gpus=-1, accelerator='ddp', max_epochs=50, log_every_n_steps=10)
        trainer.test(model=model, datamodule=data_module)

    else:
        # SMALL TEST
        # datamodule
        data_module = DIVAHisDBDataModuleCB55(data_dir='/dataset/DIVA-HisDB/segmentation/CB55-10-segmentation',
                                              crops_per_image=10, num_workers=5)

        # load the img_names_sizes information as suggested here
        # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#using-a-datamodule
        data_module.setup('test')

        model = SemanticSegmentation.load_from_checkpoint(
            '/data/usl_experiments/semantic_segmentation/supervised/baby_unet_cb55_10_crops_100/epoch=15-step=3999.ckpt',
            class_encodings=data_module.class_encodings,
            model=baby_unet(), img_names_sizes_testset=data_module.his_test.img_names_sizes)

        model.output_path = Path('/data/usl_experiments/tmp_testing_output/baby_unet_cb55_10_crops_100/')

        trainer = pl.Trainer(gpus=-1, accelerator='ddp', max_epochs=50, log_every_n_steps=10)
        trainer.test(model=model, datamodule=data_module)
