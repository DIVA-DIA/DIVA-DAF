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
    def __init__(self, data_dir: str, data_folder_name: str, gt_folder_name: str,
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 selection_test: Optional[Union[int, List[str]]] = None,
                 crop_size: int = 256, num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        super().__init__()

        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name

        analytics_data, analytics_gt = get_analytics(input_path=Path(data_dir),
                                                     data_folder_name=self.data_folder_name,
                                                     gt_folder_name=self.gt_folder_name,
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

        self.data_dir = validate_path_for_segmentation(data_dir=data_dir, data_folder_name=self.data_folder_name,
                                                       gt_folder_name=self.gt_folder_name)

        self.selection_train = selection_train
        self.selection_val = selection_val
        self.selection_test = selection_test

        self.dims = (3, crop_size, crop_size)

        # Check default attributes using base_datamodule function
        self._check_attributes()

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.train = CroppedDatasetRGB(**self._create_dataset_parameters('train'), selection=self.selection_train)
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self._check_min_num_samples(num_samples=len(self.train), data_split='train',
                                        drop_last=self.drop_last)

            self.val = CroppedDatasetRGB(**self._create_dataset_parameters('val'), selection=self.selection_val)
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self._check_min_num_samples(num_samples=len(self.val), data_split='val',
                                        drop_last=self.drop_last)

        if stage == 'test':
            self.test = CroppedDatasetRGB(**self._create_dataset_parameters('test'), selection=self.selection_test)
            log.info(f'Initialized test dataset with {len(self.test)} samples.')
            # self._check_min_num_samples(num_samples=len(self.test), data_split='test',
            #                             drop_last=False)

    def _check_min_num_samples(self, num_samples: int, data_split: str, drop_last: bool):
        num_processes = self.trainer.num_devices
        batch_size = self.batch_size
        if drop_last:
            if num_samples < (self.trainer.num_devices * self.batch_size):
                log.error(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#processes({num_processes}) times batch size ({batch_size}). '
                    f'This only works if drop_last is false!')
                raise ValueError()
        else:
            if num_samples < (self.trainer.num_devices * self.batch_size):
                log.warning(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#processes ({num_processes}) times batch size ({batch_size}). '
                    f'This works due to drop_last=False, however samples might occur multiple times. '
                    f'Check if this behavior is intended!')

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

    def get_img_name_coordinates(self, index):
        """
        Returns the original filename of the crop and its coordinate based on the index.
        You can just use this during testing!
        :param index:
        :return:
        """
        if not hasattr(self, 'test'):
            raise Exception('This method can just be called during testing')

        return self.test.img_paths_per_page[index][2:]
