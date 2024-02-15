from pathlib import Path
from typing import Optional, Callable, Union

import torch.nn as nn
import torch.optim
import torchmetrics

from src.datamodules.utils.misc import _get_argmax
from src.tasks.base_task import AbstractTask
from src.utils import utils
from src.tasks.utils.outputs import OutputKeys, reduce_dict, save_numpy_files
from tasks.utils.task_utils import print_merge_tool_info

log = utils.get_logger(__name__)


class SemanticSegmentationCroppedHisDB(AbstractTask):

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
        pixelvise semantic segmentation. The output of the network during test is a DIVAHisDB encoded image

        :param model: torch.nn.Module
            The encoder for the segmentation e.g. unet
        :param test_output_path: str
            String with a path to the output folder of the testing
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

    @staticmethod
    def to_metrics_format(x: torch.Tensor, **kwargs) -> torch.Tensor:
        return _get_argmax(x, **kwargs)

    #############################################################################################
    ########################################### TRAIN ###########################################
    #############################################################################################
    def training_step(self, batch, batch_idx, **kwargs):
        input_batch, target_batch, mask_batch = batch
        metric_kwargs = {'hisdbiou': {'mask': mask_batch}}
        output = super().training_step(batch=(input_batch, target_batch), batch_idx=batch_idx,
                                       metric_kwargs=metric_kwargs)
        return reduce_dict(input_dict=output, key_list=[OutputKeys.LOSS])

    #############################################################################################
    ############################################ VAL ############################################
    #############################################################################################

    def validation_step(self, batch, batch_idx, **kwargs):
        input_batch, target_batch, mask_batch = batch
        metric_kwargs = {'hisdbiou': {'mask': mask_batch}}
        output = super().validation_step(batch=(input_batch, target_batch), batch_idx=batch_idx,
                                         metric_kwargs=metric_kwargs)
        return reduce_dict(input_dict=output, key_list=[])

    #############################################################################################
    ########################################### TEST ############################################
    #############################################################################################

    def test_step(self, batch, batch_idx, **kwargs):
        input_batch, target_batch, mask_batch, input_idx = batch
        metric_kwargs = {'hisdbiou': {'mask': mask_batch}}
        output = super().test_step(batch=(input_batch, target_batch), batch_idx=batch_idx, metric_kwargs=metric_kwargs)

        save_numpy_files(self.trainer, self.test_output_path, input_idx, output)

        return reduce_dict(input_dict=output, key_list=[])

    def on_test_end(self) -> None:
        print_merge_tool_info(self.trainer, self.test_output_path, 'HisDB')
