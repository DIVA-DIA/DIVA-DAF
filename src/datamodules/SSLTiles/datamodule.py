from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.RotNet.utils.image_analytics import get_analytics_data
from src.datamodules.RotNet.utils.misc import validate_path_for_self_supervised
from src.datamodules.SSLTiles.datasets.dataset import DatasetSSLTiles
from src.datamodules.SSLTiles.utils.misc import GT_Type
from src.datamodules.utils.misc import get_image_dims
from src.datamodules.utils.wrapper_transforms import OnlyImage
from src.datamodules.base_datamodule import AbstractDatamodule
from src.utils import utils

log = utils.get_logger(__name__)


class SSLTilesDataModule(AbstractDatamodule):
    def __init__(self, data_dir: str, data_folder_name: str,
                 rows: int, cols: int, horizontal_shuffle: bool, vertical_shuffle: bool,
                 gt_type:str,
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 selection_test: Optional[Union[int, List[str]]] = None,
                 num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        """

        :param data_dir:
        :param data_folder_name:
        :param rows:
        :param cols:
        :param horizontal_shuffle:
        :param vertical_shuffle:
        :param gt_type: str
                The output format of the gt. (CLASSIFICATION, VECTOR, ROW_COLUMN, FULL_IMAGE)
                CLASSIFICATION: The gt is a single value ([0, row*cols-1]).
                VECTOR: The gt is a vector of length row and indicates of the row is flipped 1 or not 0. [just fpr cols=2]
                ROW_COLUMN: The gt is a metric of form cols * rows. Each tile has a unique number. e.g. [[1,0],[2,3]]
                FULL_IMAGE: The gt is the entire image. So each pixel gets the class in the ROW_COLUMN format.
        :param selection_train:
        :param selection_val:
        :param selection_test:
        :param num_workers:
        :param batch_size:
        :param shuffle:
        :param drop_last:
        """
        super().__init__()

        if gt_type not in GT_Type.__members__:
            raise ValueError(f'gt_type must be one of {GT_Type.__members__}')
        self.gt_type = GT_Type[gt_type]

        self.data_folder_name = data_folder_name
        analytics_data = get_analytics_data(input_path=Path(data_dir), data_folder_name=self.data_folder_name,
                                            get_gt_data_paths_func=DatasetSSLTiles.get_img_gt_path_list)

        self.mean = analytics_data['mean']
        self.std = analytics_data['std']
        self.class_encodings = list(range(rows * cols))
        self.num_classes = len(self.class_encodings)
        self.class_weights = torch.as_tensor([1 / self.num_classes for _ in range(self.num_classes)])

        self.image_transform = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=self.mean, std=self.std),
                                                             ]))

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.horizontal_shuffle = horizontal_shuffle
        self.vertical_shuffle = vertical_shuffle
        self.rows = rows
        self.cols = cols

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_dir = validate_path_for_self_supervised(data_dir=data_dir, data_folder_name=self.data_folder_name)

        self.selection_train = selection_train
        self.selection_val = selection_val
        self.selection_test = selection_test

        image_dims = get_image_dims(
            data_gt_path_list=DatasetSSLTiles.get_img_gt_path_list(directory=Path(data_dir) / 'train',
                                                                   data_folder_name=self.data_folder_name))
        self.image_dims = image_dims
        self.dims = (3, self.image_dims.width, self.image_dims.height)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Check default attributes using base_datamodule function
        self._check_attributes()

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.train = DatasetSSLTiles(**self._create_dataset_parameters('train'), selection=self.selection_train)
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.train),
                                       data_split='train',
                                       drop_last=self.drop_last)

            self.val = DatasetSSLTiles(**self._create_dataset_parameters('val'), selection=self.selection_val)
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.val),
                                       data_split='val',
                                       drop_last=self.drop_last)

        if stage == 'test':
            self.test = DatasetSSLTiles(**self._create_dataset_parameters('test'), selection=self.selection_test)
            log.info(f'Initialized test dataset with {len(self.test)} samples.')
            # self._check_min_num_samples(num_samples=len(self.test), data_split='test',
            #                             drop_last=False)

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

    def _create_dataset_parameters(self, dataset_type: str = 'train'):
        return {'path': self.data_dir / dataset_type,
                'data_folder_name': self.data_folder_name,
                'image_dims': self.image_dims,
                'rows': self.rows,
                'cols': self.cols,
                'horizontal_shuffle': self.horizontal_shuffle,
                'vertical_shuffle': self.vertical_shuffle,
                'image_transform': self.image_transform,
                'gt_type': self.gt_type,
                }
