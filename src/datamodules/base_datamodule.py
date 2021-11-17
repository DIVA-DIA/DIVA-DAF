from typing import Optional

import pytorch_lightning as pl
from omegaconf import OmegaConf


class AbstractDatamodule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.num_classes = -1
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
