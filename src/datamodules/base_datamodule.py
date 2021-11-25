from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


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