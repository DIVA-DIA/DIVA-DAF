from typing import Optional, Callable

import torch.nn as nn
import torch.optim
import torchmetrics

from src.tasks.base_task import AbstractTask
from src.utils import utils
from src.tasks.utils.outputs import OutputKeys, reduce_dict

log = utils.get_logger(__name__)


class Classification(AbstractTask):
    """
    Class that performs the task of classification. It overrides the methods of the base :class: `AbstractTask`.
    During all stages (training, validation, test), the model is called with the input batch and the output is
    compared with the target batch. The loss is computed and the metrics are updated.
    There are no files or folder created during testing.

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
                 confusion_matrix_val: Optional[bool] = False,
                 confusion_matrix_test: Optional[bool] = False,
                 confusion_matrix_log_every_n_epoch: Optional[int] = 1,
                 lr: float = 1e-3
                 ) -> None:
        """
        Constructor method for the :class: `classification`.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_train=metric_train,
            metric_val=metric_val,
            metric_test=metric_test,
            lr=lr,
            confusion_matrix_val=confusion_matrix_val,
            confusion_matrix_test=confusion_matrix_test,
            confusion_matrix_log_every_n_epoch=confusion_matrix_log_every_n_epoch,
        )
        # self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        super().setup(stage)

        log.info("Setup done!")

    def forward(self, x):
        return self.model(x)

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
        input_batch, target_batch = batch
        output = super().test_step(batch=(input_batch, target_batch), batch_idx=batch_idx)
        return reduce_dict(input_dict=output, key_list=[])

