from pathlib import Path
from typing import Union, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.RGB.datasets.cropped_dataset import CroppedDatasetRGB
from src.datamodules.RGB.utils.image_analytics import get_analytics
from src.datamodules.RGB.utils.single_transform import IntegerEncoding
from src.datamodules.base_datamodule import AbstractDatamodule
from src.datamodules.utils.misc import validate_path_for_segmentation
from src.datamodules.utils.twin_transforms import TwinRandomCrop
from src.datamodules.utils.wrapper_transforms import OnlyImage, OnlyTarget
from src.utils import utils

log = utils.get_logger(__name__)


class DataModuleCroppedRGB(AbstractDatamodule):
    """
    The data module for a dataset where the classes of the ground truth are encoded as colors in the image.
    This data module expects cropped with a specific structure. The cropping can be done with the script
    class: `tools/generate_cropped_dataset.py`. If you do not use the script, make sure that the images are cropped
    and named in the same way as the script does.
    If you want to work with un-cropped images use class: `DataModuleRGB`.

    The structure of the folder should be as follows::

        data_dir
        ├── data_folder_name
        │   ├── train_folder_name
        │   │   ├── original_image_name_1
        │   │   │   ├── image_crop_1.png
        │   │   │   ├── image_crop_2.png
        │   │   │   ├── ...
        │   │   │   └── image_crop_N.png
        │   ├── val_folder_name
        │   │   ├── original_image_name_1
        │   │   │   ├── image1.png
        │   │   │   ├── image2.png
        │   │   │   ├── ...
        │   │   │   └── imageN.png
        │   └── test_folder_name
        │   │   ├── original_image_name_1
        │   │   │   ├── image1.png
        │   │   │   ├── image2.png
        │   │   │   ├── ...
        │   │   │   └── imageN.png
        └── gt_folder_name
            ├── train_folder_name
            │   ├── original_image_name_1
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   ├── ...
            │   │   └── imageN.png
            ├── val_folder_name
            │   ├── original_image_name_1
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   ├── ...
            │   │   └── imageN.png
            └── test_folder_name
                ├── original_image_name_1
                │   ├── image1.png
                │   ├── image2.png
                │   ├── ...
                │   └── imageN.png

    :param data_dir: Path to the dataset folder.
    :type data_dir: str
    :param data_folder_name: Name of the folder where the images are stored.
    :type data_folder_name: str
    :param gt_folder_name: Name of the folder where the ground truth is stored.
    :type gt_folder_name: str
    :param train_folder_name: Name of the folder where the training data is stored.
    :type train_folder_name: str
    :param val_folder_name: Name of the folder where the validation data is stored.
    :type val_folder_name: str
    :param test_folder_name: Name of the folder where the test data is stored.
    :type test_folder_name: str
    :param selection_train: selection of the training data
    :type selection_train: Union[int, List[str], None]
    :param selection_val: selection of the validation data
    :type selection_val: Union[int, List[str], None]
    :param selection_test: selection of the test data
    :type selection_test: Union[int, List[str], None]
    :param num_workers: number of workers for the dataloaders
    :type num_workers: int
    :param batch_size: batch size
    :type batch_size: int
    :param shuffle: shuffle the data
    :type shuffle: bool
    :param drop_last: drop the last batch if it is smaller than the batch size
    :type drop_last: bool
    """
    def __init__(self, data_dir: str, data_folder_name: str, gt_folder_name: str,
                 train_folder_name: str = 'train', val_folder_name: str = 'val', test_folder_name: str = 'test',
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 selection_test: Optional[Union[int, List[str]]] = None,
                 crop_size: int = 256, num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        """
        Constructor method for the class: `DataModuleCroppedRGB`.
        """
        super().__init__()

        self.train_folder_name = train_folder_name
        self.val_folder_name = val_folder_name
        self.test_folder_name = test_folder_name
        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name

        analytics_data, analytics_gt = get_analytics(input_path=Path(data_dir),
                                                     data_folder_name=self.data_folder_name,
                                                     gt_folder_name=self.gt_folder_name,
                                                     train_folder_name=self.train_folder_name,
                                                     get_img_gt_path_list_func=CroppedDatasetRGB.get_gt_data_paths)

        self.mean = analytics_data['mean']
        self.std = analytics_data['std']
        self.class_encodings = analytics_gt['class_encodings']
        self.class_encodings_tensor = torch.tensor(self.class_encodings) / 255
        self.num_classes = len(self.class_encodings)
        self.class_weights = torch.as_tensor(analytics_gt['class_weights'])

        self.twin_transform = TwinRandomCrop(crop_size=crop_size)
        self.image_transform = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=self.mean, std=self.std)]))
        self.target_transform = OnlyTarget(IntegerEncoding(class_encodings=self.class_encodings_tensor))

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_dir = data_dir

        self.selection_train = selection_train
        self.selection_val = selection_val
        self.selection_test = selection_test

        self.dims = (3, crop_size, crop_size)

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.train_folder_name)
            self.train = CroppedDatasetRGB(**self._create_dataset_parameters(self.train_folder_name),
                                           selection=self.selection_train)
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.train),
                                       data_split=self.train_folder_name,
                                       drop_last=self.drop_last)

            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.val_folder_name)
            self.val = CroppedDatasetRGB(**self._create_dataset_parameters(self.val_folder_name),
                                         selection=self.selection_val)
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.val),
                                       data_split=self.val_folder_name,
                                       drop_last=self.drop_last)

        if stage == 'test':
            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.test_folder_name)
            self.test = CroppedDatasetRGB(**self._create_dataset_parameters(self.test_folder_name),
                                          selection=self.selection_test)
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
        is_test = dataset_type == 'test'
        return {'path': self.data_dir / dataset_type,
                'data_folder_name': self.data_folder_name,
                'gt_folder_name': self.gt_folder_name,
                'image_transform': self.image_transform,
                'target_transform': self.target_transform,
                'twin_transform': self.twin_transform,
                'is_test': is_test}

    def get_img_name_coordinates(self, index: int):
        """
        Returns the original filename of the crop and its coordinate based on the index.
        You can just use this during testing!

        :param index: index of the crop
        :type index: int
        :return: filename of the crop and its coordinate
        :rtype: Tuple[str, Tuple[int, int, int, int]]
        """
        if not hasattr(self, 'test'):
            raise ValueError('This method can just be called during testing')

        return self.test.img_paths_per_page[index][2:]
