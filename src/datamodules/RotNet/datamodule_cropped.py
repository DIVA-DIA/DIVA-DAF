from pathlib import Path
from typing import Union, List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.RotNet.utils.image_analytics import get_analytics_data
from src.datamodules.RotNet.datasets.cropped_dataset import CroppedRotNet
from src.datamodules.RotNet.utils.misc import validate_path_for_self_supervised
from src.datamodules.utils.wrapper_transforms import OnlyImage
from src.datamodules.base_datamodule import AbstractDatamodule
from src.utils import utils

log = utils.get_logger(__name__)


class RotNetDivaHisDBDataModuleCropped(AbstractDatamodule):
    """
    Datamodule implementation of the RoNet paper of `Gidaris et al. <https://arxiv.org/abs/1803.07728>`_. This
    datamodule is used for the DivaHisDB dataset in a cropped setup.

    The structure of the folder should be as follows::

        data_dir
        ├── train_folder_name
        │   ├── data_folder_name
        │   │   ├── original_image_name_1
        │   │   │   ├── image_crop_1.png
        │   │   │   ├── ...
        │   │   │   └── image_crop_N.png
        │   │   └──original_image_name_N
        │   │       ├── image_crop_1.png
        │   │       ├── ...
        │   │       └── image_crop_N.png
        │   └── gt_folder_name
        │       ├── original_image_name_1
        │       │   ├── image_crop_1.png
        │       │   ├── ...
        │       │   └── image_crop_N.png
        │       └──original_image_name_N
        │           ├── image_crop_1.png
        │           ├── ...
        │           └── image_crop_N.png
        ├── validation_folder_name
        │   ├── data_folder_name
        │   │   ├── original_image_name_1
        │   │   │   ├── image_crop_1.png
        │   │   │   ├── ...
        │   │   │   └── image_crop_N.png
        │   │   └──original_image_name_N
        │   │       ├── image_crop_1.png
        │   │       ├── ...
        │   │       └── image_crop_N.png
        │   └── gt_folder_name
        │       ├── original_image_name_1
        │       │   ├── image_crop_1.png
        │       │   ├── ...
        │       │   └── image_crop_N.png
        │       └──original_image_name_N
        │           ├── image_crop_1.png
        │           ├── ...
        │           └── image_crop_N.png
        └── test_folder_name
            ├── data_folder_name
            │   ├── original_image_name_1
            │   │   ├── image_crop_1.png
            │   │   ├── ...
            │   │   └── image_crop_N.png
            │   └──original_image_name_N
            │       ├── image_crop_1.png
            │       ├── ...
            │       └── image_crop_N.png
            └── gt_folder_name
                ├── original_image_name_1
                │   ├── image_crop_1.png
                │   ├── ...
                │   └── image_crop_N.png
                └──original_image_name_N
                    ├── image_crop_1.png
                    ├── ...
                    └── image_crop_N.png

    :param data_dir: Path to root dir of the dataset (folder containing the train/val/test folder)
    :type data_dir: str
    :param data_folder_name: Name of the folder containing the train/val/test folder
    :type data_folder_name: str
    :param selection_train: Selection of the train set. Can be either a list of strings or an integer. If it is a list
                            of strings, it should contain the names of the images to be used. If it is an integer, it
                            should be the number of images to be used. If None, all images are used.
    :type selection_train: Optional[Union[int, List[str]]]
    :param selection_val: Selection of the validation set. Can be either a list of strings or an integer. If it is a
                            list of strings, it should contain the names of the images to be used. If it is an integer,
                            it should be the number of images to be used. If None, all images are used.
    :type selection_val: Optional[Union[int, List[str]]]
    :param selection_test: Selection of the test set. Can be either a list of strings or an integer. If it is a list
                            of strings, it should contain the names of the images to be used. If it is an integer, it
                            should be the number of images to be used. If None, all images are used.
    :type selection_test: Optional[Union[int, List[str]]]
    :param crop_size: Size of the crop to be used
    :type crop_size: int
    :param num_workers: Number of workers to be used for loading the data
    :type num_workers: int
    :param batch_size: Batch size to be used
    :type batch_size: int
    :param shuffle: Whether to shuffle the data
    :type shuffle: bool
    :param drop_last: Whether to drop the last batch
    :type drop_last: bool
    """
    def __init__(self, data_dir: str, data_folder_name: str,
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 selection_test: Optional[Union[int, List[str]]] = None,
                 crop_size: int = 256, num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True, ):
        """
        Constructor method for the RotNetDivaHisDBDataModuleCropped class.
        """
        super().__init__()

        self.data_folder_name = data_folder_name
        analytics_data = get_analytics_data(input_path=Path(data_dir), data_folder_name=self.data_folder_name,
                                            get_gt_data_paths_func=CroppedRotNet.get_gt_data_paths)

        self.mean = analytics_data['mean']
        self.std = analytics_data['std']
        self.class_encodings = np.array([0, 90, 180, 270])
        self.num_classes = len(self.class_encodings)
        self.class_weights = torch.as_tensor([1 / self.num_classes for _ in range(self.num_classes)])

        self.image_transform = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=self.mean, std=self.std),
                                                             transforms.RandomCrop(size=crop_size)]))

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_dir = validate_path_for_self_supervised(data_dir=data_dir, data_folder_name=self.data_folder_name)

        self.selection_train = selection_train
        self.selection_val = selection_val
        self.selection_test = selection_test

        self.dims = (3, crop_size, crop_size)

        # Check default attributes using base_datamodule function
        self._check_attributes()

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.train = CroppedRotNet(**self._create_dataset_parameters('train'), selection=self.selection_train)
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.train),
                                       data_split='train',
                                       drop_last=self.drop_last)

            self.val = CroppedRotNet(**self._create_dataset_parameters('val'), selection=self.selection_val)
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.val),
                                       data_split='val',
                                       drop_last=self.drop_last)

        if stage == 'test':
            self.test = CroppedRotNet(**self._create_dataset_parameters('test'), selection=self.selection_test)
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

    def _create_dataset_parameters(self, dataset_type: str = 'train') -> Dict[str, Any]:
        """
        Creates the parameters for the dataset that are common for all splits.

        :param dataset_type: Type of the dataset (train/val/test)
        :type dataset_type: str
        :return: Dictionary containing the parameters for the dataset
        :rtype: Dict[str, Any]
        """
        is_test = dataset_type == 'test'
        return {'path': self.data_dir / dataset_type,
                'data_folder_name': self.data_folder_name,
                'image_transform': self.image_transform,
                'is_test': is_test}
