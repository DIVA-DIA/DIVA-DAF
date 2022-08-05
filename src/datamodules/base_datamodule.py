from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.utils import utils

log = utils.get_logger(__name__)


class AbstractDatamodule(pl.LightningDataModule):
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
        assert self.num_classes > 0
        if self.class_weights is not None:
            assert len(self.class_weights) == self.num_classes
            assert torch.is_tensor(self.class_weights)

    @staticmethod
    def check_min_num_samples(num_devices, batch_size1, num_samples: int, data_split: str, drop_last: bool):
        batch_size = batch_size1
        if drop_last:
            if num_samples < (num_devices * batch_size1):
                log.error(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#devices({num_devices}) times batch size ({batch_size}). '
                    f'This only works if drop_last is false!')
                raise ValueError()
        else:
            if num_samples < (num_devices * batch_size1):
                log.warning(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#devices ({num_devices}) times batch size ({batch_size}). '
                    f'This works due to drop_last=False, however samples might occur multiple times. '
                    f'Check if this behavior is intended!')
