from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.DivaHisDB.utils.single_transform import IntegerEncoding
from src.datamodules.base_datamodule import AbstractDatamodule
from src.datamodules.DivaHisDB.datasets.cropped_dataset import CroppedHisDBDataset
from src.datamodules.DivaHisDB.utils.image_analytics import get_analytics
from src.datamodules.utils.misc import validate_path_for_segmentation
from src.datamodules.utils.twin_transforms import TwinRandomCrop
from src.datamodules.utils.wrapper_transforms import OnlyImage, OnlyTarget
from src.utils import utils

log = utils.get_logger(__name__)


class DivaHisDBDataModuleCropped(AbstractDatamodule):
    """
    DataModule for the DivaHisDB dataset or a similar dataset with the same folder structure and ground truth encoding.

    The ground truth encoding is like the following:
    Red = 0 everywhere (except boundaries)
    Green = 0 everywhere

    Blue = 0b00...1000 = 0x000008: main text body
    Blue = 0b00...0100 = 0x000004: decoration
    Blue = 0b00...0010 = 0x000002: comment
    Blue = 0b00...0001 = 0x000001: background (out of page)

    Blue = 0b...1000 | 0b...0010 = 0b...1010 = 0x00000A : main text body + comment
    Blue = 0b...1000 | 0b...0100 = 0b...1100 = 0x00000C : main text body + decoration
    Blue = 0b...0010 | 0b...0100 = 0b...0110 = 0x000006 : comment + decoration

    The structure of the folder should be as follows:

    data_dir
    ├── data_folder_name
    │   ├── train_folder_name
    │   │   ├── image1.png
    │   │   ├── image2.png
    │   │   ├── ...
    │   │   └── imageN.png
    │   ├── val_folder_name
    │   │   ├── image1.png
    │   │   ├── image2.png
    │   │   ├── ...
    │   │   └── imageN.png
    │   └── test_folder_name
    │       ├── image1.png
    │       ├── image2.png
    │       ├── ...
    │       └── imageN.png
    └── gt_folder_name
        ├── train_folder_name
        │   ├── image1.png
        │   ├── image2.png
        │   ├── ...
        │   └── imageN.png
        ├── val_folder_name
        │   ├── image1.png
        │   ├── image2.png
        │   ├── ...
        │   └── imageN.png
        └── test_folder_name
            ├── image1.png
            ├── image2.png
            ├── ...
            └── imageN.png

    :param data_dir: path to the data directory
    :type data_dir: str
    :param data_folder_name: name of the folder containing the images
    :type data_folder_name: str
    :param gt_folder_name: name of the folder containing the ground truth
    :type gt_folder_name: str
    :param train_folder_name: name of the folder containing the training data
    :type train_folder_name: str
    :param val_folder_name: name of the folder containing the validation data
    :type val_folder_name: str
    :param test_folder_name: name of the folder containing the test data
    :type test_folder_name: str
    :param selection_train: selection of the training data
    :type selection_train: Union[int, List[str], None]
    :param selection_val: selection of the validation data
    :type selection_val: Union[int, List[str], None]
    :param selection_test: selection of the test data
    :type selection_test: Union[int, List[str], None]
    :param crop_size: size of the crops
    :type crop_size: int
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
                 selection_train: Optional[Union[int, List[str], None]] = None,
                 selection_val: Optional[Union[int, List[str], None]] = None,
                 selection_test: Optional[Union[int, List[str], None]] = None,
                 crop_size: int = 256, num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True) -> None:
        """
        Constructor of the DivaHisDBDataModuleCropped class.
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
                                                     get_gt_data_paths_func=CroppedHisDBDataset.get_gt_data_paths)

        self.mean = analytics_data['mean']
        self.std = analytics_data['std']
        self.class_encodings = analytics_gt['class_encodings']
        self.num_classes = len(self.class_encodings)
        self.class_weights = torch.as_tensor(analytics_gt['class_weights'])

        self.twin_transform = TwinRandomCrop(crop_size=crop_size)
        self.image_transform = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=self.mean, std=self.std)]))
        self.target_transform = OnlyTarget(IntegerEncoding(class_encodings=self.class_encodings))

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name

        self.data_dir = data_dir

        self.selection_train = selection_train
        self.selection_val = selection_val
        self.selection_test = selection_test

        self.dims = (3, crop_size, crop_size)

        # Check default attributes using base_datamodule function
        self._check_attributes()

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup()
        if stage == 'fit' or stage is None:
            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.train_folder_name)
            self.train = CroppedHisDBDataset(**self._create_dataset_parameters('train'), selection=self.selection_train)
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.train),
                                       data_split=self.train_folder_name,
                                       drop_last=self.drop_last)

            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.val_folder_name)
            self.val = CroppedHisDBDataset(**self._create_dataset_parameters('val'), selection=self.selection_val)
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.val),
                                       data_split=self.val_folder_name,
                                       drop_last=self.drop_last)

        if stage == 'test':
            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.test_folder_name)
            self.test = CroppedHisDBDataset(**self._create_dataset_parameters('test'), selection=self.selection_test)
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
        is_test = dataset_type == 'test'
        return {'path': self.data_dir / dataset_type,
                'data_folder_name': self.data_folder_name,
                'gt_folder_name': self.gt_folder_name,
                'image_transform': self.image_transform,
                'target_transform': self.target_transform,
                'twin_transform': self.twin_transform,
                'is_test': is_test}

    def get_img_name_coordinates(self, index) -> Tuple[Path, Path, str, str, Tuple[int, int]]:
        """
        Returns the original filename of the crop and its coordinate based on the index.
        You can just use this during testing!

        :param index: index of the crop
        :type index: int
        :return: filename, x, y
        """
        if not hasattr(self, 'test'):
            raise Exception('This method can just be called during testing')

        return self.test.img_paths_per_page[index][2:]
