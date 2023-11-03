from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.utils import utils

log = utils.get_logger(__name__)


class AbstractDatamodule(pl.LightningDataModule):
    """
    Abstract class for all datamodules. All datamodules should inherit from this class.
    It provides some basic functionality like checking the number of samples and the number of classes.
    Also, it provides a resolver for the datamodule object itself, so that it can be used in the config.
    The class variable `dims` must be set in the subclass.
    """

    def __init__(self):
        super().__init__()
        self.num_classes = -1
        self.class_weights = None
        resolver_name = 'datamodule'
        if not OmegaConf.has_resolver(resolver_name):
            OmegaConf.register_new_resolver(
                resolver_name,
                lambda name: getattr(self, name),
                use_cache=False
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.dims:
            raise ValueError("the dimensions of the data needs to be set! self.dims")

    def _check_attributes(self):
        """
        Checks if all attributes are set correctly.

        :raises ValueError: If the attributes are not set correctly
        """
        assert self.num_classes > 0
        if self.class_weights is not None:
            assert len(self.class_weights) == self.num_classes
            assert torch.is_tensor(self.class_weights)

    @staticmethod
    def check_min_num_samples(num_devices: int, batch_size_input: int, num_samples: int, data_split: str,
                              drop_last: bool):
        """
        Checks if the number of samples is sufficient for the given batch size and number of devices.
        
        :param num_devices: The number of devices
        :type num_devices: int
        :param batch_size_input: The batch size
        :type batch_size_input: int
        :param num_samples: The number of samples
        :type num_samples: int
        :param data_split: The data split (train, val, test)
        :type data_split: str
        :param drop_last: Whether to drop the last batch if it is smaller than the batch size
        :type drop_last: bool
        :raises ValueError: If the number of samples is not sufficient
        """
        batch_size = batch_size_input
        if drop_last:
            if num_samples < (num_devices * batch_size_input):
                log.error(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#devices({num_devices}) times batch size ({batch_size}). '
                    f'This only works if drop_last is false!')
                raise ValueError()
        else:
            if num_samples < (num_devices * batch_size_input):
                log.warning(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#devices ({num_devices}) times batch size ({batch_size}). '
                    f'This works due to drop_last=False, however samples might occur multiple times. '
                    f'Check if this behavior is intended!')
