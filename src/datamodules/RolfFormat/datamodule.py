from pathlib import Path
from typing import Union, List, Optional

import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.RolfFormat.datasets.dataset import DatasetRolfFormat
from src.datamodules.RolfFormat.utils.image_analytics import get_analytics_data, get_analytics_gt
from src.datamodules.RolfFormat.utils.twin_transforms import IntegerEncoding
from src.datamodules.RolfFormat.utils.wrapper_transforms import OnlyImage, OnlyTarget
from src.datamodules.base_datamodule import AbstractDatamodule
from src.utils import utils

log = utils.get_logger(__name__)

@dataclass
class DatasetSpecs:
    data_root: str
    doc_dir: str
    doc_names: str
    gt_dir: str
    gt_names: str
    range_from: int
    range_to: int


class DataModuleRolfFormat(AbstractDatamodule):
    def __init__(self, data_root: str,
                 train_specs=None, val_specs=None, test_specs=None,
                 image_analytics=None, classes=None, image_dims=None,
                 num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        super().__init__()

        train_dataset_specs = [DatasetSpecs(data_root=data_root, **v) for k, v in train_specs.items()]
        val_dataset_specs = [DatasetSpecs(data_root=data_root, **v) for k, v in val_specs.items()]
        test_dataset_specs = [DatasetSpecs(data_root=data_root, **v) for k, v in test_specs.items()]

        analytics_data = get_analytics_data()
        analytics_gt = get_analytics_gt()

        self.mean = analytics_data['mean']
        self.std = analytics_data['std']
        self.class_encodings = analytics_gt['class_encodings']
        self.class_encodings_tensor = torch.tensor(self.class_encodings) / 255
        self.num_classes = len(self.class_encodings)
        self.class_weights = analytics_gt['class_weights']

        self.twin_transform = None
        self.image_transform = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=self.mean, std=self.std)]))
        self.target_transform = OnlyTarget(IntegerEncoding(class_encodings=self.class_encodings_tensor))

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.dims = (3, image_dims['width'], image_dims['height'])

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.train = DatasetRolfFormat(**self._create_dataset_parameters('train'), selection=self.selection_train)
            self.val = DatasetRolfFormat(**self._create_dataset_parameters('val'), selection=self.selection_val)

            self._check_min_num_samples(num_samples=len(self.train), data_split='train',
                                        drop_last=self.drop_last)
            self._check_min_num_samples(num_samples=len(self.val), data_split='val',
                                        drop_last=self.drop_last)

        if stage == 'test' or stage is not None:
            self.test = DatasetRolfFormat(**self._create_dataset_parameters('test'), selection=self.selection_test)
            # self._check_min_num_samples(num_samples=len(self.test), data_split='test',
            #                             drop_last=False)

    def _check_min_num_samples(self, num_samples: int, data_split: str, drop_last: bool):
        num_processes = self.trainer.num_processes
        batch_size = self.batch_size
        if drop_last:
            if num_samples < (self.trainer.num_processes * self.batch_size):
                log.error(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#processes({num_processes}) times batch size ({batch_size}). '
                    f'This only works if drop_last is false!')
                raise ValueError()
        else:
            if num_samples < (self.trainer.num_processes * self.batch_size):
                log.warning(
                    f'#samples ({num_samples}) in "{data_split}" smaller than '
                    f'#processes ({num_processes}) times batch size ({batch_size}). '
                    f'This works due to drop_last=False, however samples will occur multiple times. '
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
                'classes': self.class_encodings,
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
