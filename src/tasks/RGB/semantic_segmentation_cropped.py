from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import torch.nn as nn
import torch.optim
import torchmetrics

from src.datamodules.utils.misc import _get_argmax
from src.tasks.base_task import AbstractTask
from src.utils import utils
from src.tasks.utils.outputs import OutputKeys, reduce_dict

log = utils.get_logger(__name__)


class SemanticSegmentationCroppedRGB(AbstractTask):
    """
    Semantic Segmentation task for cropped images that are RGB encoded, so the class is encoded in the color.
    The output for the test are also patches that can be stitched together with the :class: `CroppedOutputMergerRGB`
    and are in the RGB format as well as raw prediction of the network in numpy format.

    :param model: The model to train, validate and test.
    :type model: nn.Module
    :param optimizer: The optimizer used during training.
    :type optimizer: torch.optim.Optimizer
    :param loss_fn: The loss function used during training, validation, and testing.
    :type loss_fn: Callable
    :param metric_train: The metric used during training.
    :type metric_train: torchmetrics.Metric
    :param metric_val: The metric used during validation.
    :type metric_val: torchmetrics.Metric
    :param metric_test: The metric used during testing.
    :type metric_test: torchmetrics.Metric
    :param confusion_matrix_val: Whether to compute the confusion matrix during validation.
    :type confusion_matrix_val: bool
    :param confusion_matrix_test: Whether to compute the confusion matrix during testing.
    :type confusion_matrix_test: bool
    :param confusion_matrix_log_every_n_epoch: The frequency of logging the confusion matrix.
    :type confusion_matrix_log_every_n_epoch: int
    :param lr: The learning rate.
    :type lr: float
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Optional[Callable] = None,
                 metric_train: Optional[torchmetrics.Metric] = None,
                 metric_val: Optional[torchmetrics.Metric] = None,
                 metric_test: Optional[torchmetrics.Metric] = None,
                 test_output_path: Optional[Union[str, Path]] = 'test_output',
                 predict_output_path: Optional[Union[str, Path]] = 'predict_output',
                 confusion_matrix_val: Optional[bool] = False,
                 confusion_matrix_test: Optional[bool] = False,
                 confusion_matrix_log_every_n_epoch: Optional[int] = 1,
                 lr: float = 1e-3
                 ) -> None:
        """
        Construction method for RGB SegemntationCropped task.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_train=metric_train,
            metric_val=metric_val,
            metric_test=metric_test,
            test_output_path=test_output_path,
            predict_output_path=predict_output_path,
            lr=lr,
            confusion_matrix_val=confusion_matrix_val,
            confusion_matrix_test=confusion_matrix_test,
            confusion_matrix_log_every_n_epoch=confusion_matrix_log_every_n_epoch,
        )
        # self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        super().setup(stage)

        if not hasattr(self.trainer.datamodule, 'get_img_name_coordinates'):
            raise NotImplementedError('DataModule needs to implement get_img_name_coordinates function')

        log.info("Setup done!")

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def to_metrics_format(x: torch.Tensor, **kwargs) -> torch.Tensor:
        return _get_argmax(x, **kwargs)

    #############################################################################################
    ########################################### TRAIN ###########################################
    #############################################################################################
    def training_step(self, batch, batch_idx, **kwargs):
        input_batch, target_batch = batch
        output = super().training_step(batch=(input_batch, target_batch), batch_idx=batch_idx)
        return reduce_dict(input_dict=output, key_list=[OutputKeys.LOSS])

    #############################################################################################
    ############################################ VAL ############################################
    #############################################################################################

    def validation_step(self, batch, batch_idx, **kwargs):
        input_batch, target_batch = batch
        output = super().validation_step(batch=(input_batch, target_batch), batch_idx=batch_idx)
        return reduce_dict(input_dict=output, key_list=[])

    #############################################################################################
    ########################################### TEST ############################################
    #############################################################################################

    def test_step(self, batch, batch_idx, **kwargs):
        input_batch, target_batch, input_idx = batch
        output = super().test_step(batch=(input_batch, target_batch), batch_idx=batch_idx)

        if not hasattr(self.trainer.datamodule, 'get_img_name_coordinates'):
            raise NotImplementedError('Datamodule does not provide detailed information of the crop')

        for patch, idx in zip(output[OutputKeys.PREDICTION].detach().cpu().numpy(),
                              input_idx.detach().cpu().numpy()):
            patch_info = self.trainer.datamodule.get_img_name_coordinates(idx)
            img_name = patch_info[0]
            patch_name = patch_info[1]
            dest_folder = self.test_output_path / 'patches' / img_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_filename = dest_folder / f'{patch_name}.npy'

            np.save(file=str(dest_filename), arr=patch)

        return reduce_dict(input_dict=output, key_list=[])

    def on_test_end(self) -> None:
        datamodule_path = self.trainer.datamodule.data_dir
        prediction_path = (self.test_output_path / 'patches').absolute()
        output_path = (self.test_output_path / 'result').absolute()

        data_folder_name = self.trainer.datamodule.data_folder_name
        gt_folder_name = self.trainer.datamodule.gt_folder_name

        log.info(f'To run the merging of patches:')
        log.info(f'python tools/merge_cropped_output_RGB.py -d {datamodule_path} -p {prediction_path} -o {output_path} '
                 f'-df {data_folder_name} -gf {gt_folder_name}')
