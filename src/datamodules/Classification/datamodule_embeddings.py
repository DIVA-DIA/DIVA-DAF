import json
from pathlib import Path
from typing import Union, List, Optional, Dict, Callable

from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.Classification.dataset.embedding_dataset import EmbeddingsDataset
from src.datamodules.base_datamodule import AbstractDatamodule
from src.utils import utils

log = utils.get_logger(__name__)


class EmbeddingClassificationDatamodule(AbstractDatamodule):

    def __init__(self, data_dir: str,
                 embedding: str = "aws_text-embedding-3-small",
                 num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        """
        Constructor method for the ClassificationDatamodule class.
        """
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_dir = Path(data_dir)
        self.embedding = embedding

        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.num_classes = len(self.classes)

        data = filter(lambda s: 'train' in str(s) and self.embedding in str(s), self.data_dir.iterdir())
        self.embedding_size = json.load(next(data).open('r'))['embedding_size']
        self.dims = self.embedding_size

        self.train = None
        self.val = None
        self.test = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.train = EmbeddingsDataset(**self._create_dataset_parameters('train'))
            log.info(f'Initialized train dataset with {len(self.train)} samples.')

            self.val = EmbeddingsDataset(**self._create_dataset_parameters('val'))
            log.info(f'Initialized val dataset with {len(self.val)} samples.')

        if stage == 'test':
            self.test = EmbeddingsDataset(**self._create_dataset_parameters('val'))
            log.info(f'Initialized val dataset with {len(self.val)} samples.')

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

    def _create_dataset_parameters(self, dataset_type: str = 'train') -> Dict[str, Union[Path, Callable]]:
        """
        Creates the parameters for the ImageFolder dataset.

        :param dataset_type: Type of the dataset. Either 'train', 'val' or 'test'.
        :type dataset_type: str
        :return: Parameters for the ImageFolder dataset.
        :rtype: Dict[str, Union[Path, Callable]]
        """

        return {'root': self.data_dir,
                'split': dataset_type,
                'embedding': self.embedding,
                }
