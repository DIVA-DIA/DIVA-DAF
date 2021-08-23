import io
import re
from pathlib import Path
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch.optim
import wandb
from PIL import Image
from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCropped
from matplotlib.patches import Rectangle
from pl_bolts.models.vision import UNet
from pytorch_lightning import seed_everything
from torch.nn import Module, functional as F

from src.tasks.semantic_segmentation.utils.accuracy import accuracy_segmentation
from src.tasks.semantic_segmentation.utils.output_tools import save_output_page_image, merge_patches, _get_argmax


class SemanticSegmentation(pl.LightningModule):

    def __init__(self,
                 # datamodule: pl.LightningDataModule,
                 model: Module, optimizer: torch.optim.Optimizer,
                 output_path: str = 'test_images', create_confusion_matrix=False,
                 calc_his_miou_train_val=False, calc_his_miou_test=False):
        """
        pixelvise semantic segmentation. The output of the network during test is a DIVAHisDB encoded image

        :param model: torch.nn.Module
            The encoder for the segmentation e.g. unet
        :param class_encodings: list(int)
            A list of the class encodings so the classes as integers (e.g. [1, 2, 4, 8])
        :param output_path: str
            String with a path to the output folder of the testing
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        # this is the dictionary to collect the different crops during testing
        self.canvas = {}
        # list of tuples -> (gt_img_name, img_size (H, W))

        self.class_encodings = []
        self.num_classes = -1

        self.create_confusion_matrix = create_confusion_matrix
        self.calc_his_miou_train_val = calc_his_miou_train_val
        self.calc_his_miou_test = calc_his_miou_test

        # paths
        self.output_path = Path(output_path)  # / f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.create_confusion_matrix:
            self.output_path_analysis = self.output_path / 'analysis'
            self.train_cm = None
            self.val_cm = None
            self.test_cm = None

    def setup(self, stage: str) -> None:
        super().setup(stage)

        if not hasattr(self.trainer.datamodule, 'get_img_name_coordinates'):
            raise NotImplementedError('DataModule needs to implement get_img_name_coordinates function')

        self.class_encodings = self.trainer.datamodule.class_encodings
        self.num_classes = len(self.trainer.datamodule.class_encodings)

        # metrics
        if self.create_confusion_matrix:
            self.output_path_analysis.mkdir(parents=True, exist_ok=True)
            self.train_cm = pl.metrics.ConfusionMatrix(num_classes=self.num_classes, normalize='true')
            self.val_cm = pl.metrics.ConfusionMatrix(num_classes=self.num_classes, normalize='true')
            self.test_cm = pl.metrics.ConfusionMatrix(num_classes=self.num_classes, normalize='true')

        print("Setup done!")

    def forward(self, x):
        return self.model(x)

    #############################################################################################
    ########################################### TRAIN ###########################################
    #############################################################################################
    def training_step(self, batch, batch_idx, **kwargs):
        return_dict = {}
        # TODO assert check that the target_raw has len 2 (gt and mask)
        # get image and gt
        img, target, mask = batch
        y_hat = self.model(img)
        loss = F.cross_entropy(y_hat, target)
        return_dict['loss'] = loss

        # takes from the prediction array the highest value like in the gt
        y_hat_encoded = _get_argmax(y_hat)

        if self.create_confusion_matrix:
            self.train_cm(y_hat_encoded, target)

        # Metric Logging
        self.log('train/loss', loss, on_epoch=True)
        if self.calc_his_miou_train_val:
            _, _, his_miou, _ = accuracy_segmentation(label_trues=target,
                                                      label_preds=y_hat_encoded,
                                                      # n_class=self.num_classes,
                                                      n_class=len(self.trainer.datamodule.class_encodings),
                                                      mask=mask,
                                                      calc_mean_iu=True)
            self.log('train/iou', his_miou, on_epoch=True)

        return return_dict

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.create_confusion_matrix:
            self._create_and_save_conf_mat(cm=self.train_cm, status='train')
            self.train_cm.reset()

    #############################################################################################
    ############################################ VAL ############################################
    #############################################################################################

    def validation_step(self, batch, batch_idx):
        return_dict = {}
        img, target, mask = batch
        y_hat = self.model(img)
        return_dict['targets'] = target
        loss = F.cross_entropy(y_hat, target)
        return_dict['loss'] = loss

        # takes from the prediction array the highest value like in the gt
        y_hat_encoded = _get_argmax(y_hat)
        return_dict['preds'] = y_hat_encoded

        if self.create_confusion_matrix:
            self.val_cm(y_hat_encoded, target)

        # Metric Logging
        self.log('val/loss', loss, on_epoch=True)
        if self.calc_his_miou_train_val:
            _, _, his_miou, _ = accuracy_segmentation(label_trues=target,
                                                      label_preds=y_hat_encoded,
                                                      # n_class=self.num_classes,
                                                      n_class=len(self.trainer.datamodule.class_encodings),
                                                      mask=mask,
                                                      calc_mean_iu=True)
            self.log('val/iou', his_miou, on_epoch=True)

        return return_dict

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        if self.create_confusion_matrix:
            # y_hat_epoch = cat([batch['y_hat'] for batch in outputs])
            # target_epoch = cat([batch['target'] for batch in outputs])
            self._create_and_save_conf_mat(cm=self.val_cm, status='val')
            self.val_cm.reset()

    #############################################################################################
    ########################################### TEST ############################################
    #############################################################################################

    def test_step(self, batch, batch_idx, **kwargs):
        return_dict = {}
        input_batch, target, mask, input_idx = batch
        y_hat = self.model(input_batch)
        loss = F.cross_entropy(y_hat, target)
        return_dict['loss'] = loss

        # takes from the prediction array the highest value like in the gt
        y_hat_encoded = _get_argmax(y_hat)

        if self.create_confusion_matrix:
            self.test_cm(y_hat_encoded, target)

        # Metric Logging
        self.log('test/loss', loss, on_epoch=True, on_step=True)
        if self.calc_his_miou_test:
            _, _, his_miou, _ = accuracy_segmentation(label_trues=target,
                                                      label_preds=y_hat_encoded,
                                                      n_class=self.num_classes,
                                                      mask=mask,
                                                      calc_mean_iu=True)
            self.log('test/iou', his_miou, on_epoch=True, on_step=True)

        if not hasattr(self.trainer.datamodule, 'get_img_name_coordinates'):
            raise NotImplementedError('Datamodule does not provide detailed information of the crop')

        for patch, idx in zip(y_hat.data.detach().cpu().numpy(),
                              input_idx.detach().cpu().numpy()):
            patch_info = self.trainer.datamodule.get_img_name_coordinates(idx)
            img_name = patch_info[0]
            patch_name = patch_info[1]
            dest_folder = self.output_path / 'patches' / img_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_filename = dest_folder / f'{patch_name}.npy'

            np.save(file=str(dest_filename), arr=patch)

        return return_dict

    def test_epoch_end(self, outputs: List[Any]) -> None:
        if self.create_confusion_matrix:
            # y_hat_epoch = cat([batch['y_hat'] for batch in outputs])
            # target_epoch = cat([batch['target'] for batch in outputs])
            self._create_and_save_conf_mat(cm=self.test_cm, status='test')
            self.test_cm.reset()

    def on_test_epoch_end(self) -> None:
        pass
        # # Merge patches on canvas
        # img_name_list = [n for n in self.output_path.iterdir() if n.is_dir()]
        # for img_name in img_name_list:
        #     patches_folder = self.output_path / 'patches' / img_name
        #     coordinates = re.compile(r'.+_x(\d+)_y(\d+)\.npy$')
        #
        #     if not patches_folder.is_dir():
        #         continue
        #
        #     patches_list = []
        #     for patch_file in patches_folder.glob(f'{img_name}*.npy'):
        #         m = coordinates.match(patch_file.name)
        #         if m is None:
        #             continue
        #         x = int(m.group(1))
        #         y = int(m.group(2))
        #         patch = np.load(str(patch_file))
        #         patches_list.append((patch, x, y))
        #     patches_list = sorted(patches_list, key=lambda v: (v[2], v[1]))
        #
        #     # Create new canvas
        #     canvas = np.empty((self.num_classes, *img_size_dict[img_name]))
        #     canvas.fill(np.nan)
        #
        #     for patch, x, y in patches_list:
        #         # Add the patch to the image
        #         canvas = merge_patches(patch, (x, y), canvas)
        #
        #     # Save the image when done
        #     if not np.isnan(np.sum(canvas)):
        #         # Save the final image (image_name, output_image, output_folder, class_encoding)
        #         save_output_page_image(image_name=img_name, output_image=canvas,
        #                                output_folder=self.output_path,
        #                                class_encoding=self.trainer.datamodule.class_encodings)
        #                                # class_encoding=self.class_encodings)
        #     else:
        #         print(f'WARNING: Test image {img_name} was not written! It still contains NaN values.')

    def configure_optimizers(self):
        return self.hparams.optimizer

    def _create_and_save_conf_mat(self, cm, status: str = 'train'):
        conf_mat_name = f'{status}_CM_epoch_{self.current_epoch}'
        df_cm = pd.DataFrame(cm.compute().cpu().numpy())
        plt.figure(figsize=(11, 9))
        # create the cm
        fig = sn.heatmap(df_cm, annot=True, square=True)
        # highlight diagonal
        for i in range(self.num_classes):
            fig.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=3))
        plt.xlabel('Predictions')
        plt.ylabel('Targets')
        plt.title(conf_mat_name)
        # save plot as normal image on the disc
        plt.savefig(fname=self.output_path_analysis / (conf_mat_name + '.png'), format='png')
        # getting the plot as pil image according to
        # https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
        if pl.loggers.WandbLogger is self.logger.__class__:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            self.logger.experiment.log({conf_mat_name: [wandb.Image(img)]}, commit=False)
            buf.close()
        plt.close()


if __name__ == '__main__':
    # because of ddp
    seed_everything(42)
    # datamodule
    # data_module = DIVAHisDBDataModuleCropped(
    #     data_dir='/data/usl_experiments/semantic_segmentation/datasets_cropped/CB55',
    #     batch_size=16, num_workers=4)

    data_module = DIVAHisDBDataModuleCropped(
        data_dir='/data/usl_experiments/semantic_segmentation/datasets_cropped/CB55-10-segmentation',
        batch_size=16, num_workers=4)

    # import pickle
    # with open('/data/usl_experiments/tmp_testing_output/temp_v2021_04_21.pkl', 'wb') as f:
    #     pickle.dump(data_module, f)

    # # load the img_names_sizes information as suggested here
    # # https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#using-a-datamodule
    # data_module.setup('test')

    def baby_unet():
        return UNet(num_classes=len(data_module.class_encodings), num_layers=2, features_start=32)


    # segmentation = SemanticSegmentation(model=UNet(num_classes=len(data_module.class_encodings)),
    #                                     datamodule=data_module,
    #                                     # create_confusion_matrix=True,
    #                                     output_path='/data/usl_experiments/tmp_testing_output/unet_cropped_cb55_v2021_04_21/')

    segmentation = SemanticSegmentation(model=baby_unet(),
                                        # datamodule=data_module,
                                        # create_confusion_matrix=True,
                                        output_path='/data/usl_experiments/tmp_testing_output/baby_unet_cropped_cb55_v2021_04_22a/',
                                        create_confusion_matrix=True,
                                        calc_his_miou_train_val=True,
                                        calc_his_miou_test=True
                                        )

    # logger
    wandb_logger = pl.loggers.WandbLogger(name="baby_unet_cropped_cb55_v2021_04_22a", project='unsupervised',
                                          # log_model=True, save_dir=segmentation.output_path
                                          )

    # checkpoint requirements
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', dirpath=segmentation.output_path)

    trainer = pl.Trainer(gpus=-1,
                         accelerator='ddp',
                         max_epochs=5,
                         log_every_n_steps=10,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback])

    trainer.fit(model=segmentation, datamodule=data_module)

    # trainer.test(model=segmentation)
