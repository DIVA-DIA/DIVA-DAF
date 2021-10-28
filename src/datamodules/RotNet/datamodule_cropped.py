from pathlib import Path
from typing import Union, List, Optional

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.RotNet.utils.image_analytics import get_analytics
from src.datamodules.RotNet.datasets.cropped_dataset import CroppedRotNet, ROTATION_ANGLES
from src.datamodules.RotNet.utils.misc import validate_path_for_self_supervised
from src.datamodules.RotNet.utils.wrapper_transforms import OnlyImage
from src.datamodules.base_datamodule import AbstractDatamodule
from src.utils import utils

log = utils.get_logger(__name__)


class RotNetDivaHisDBDataModuleCropped(AbstractDatamodule):
    def __init__(self, data_dir: str = None, data_folder_name: str = 'data',
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 selection_test: Optional[Union[int, List[str]]] = None,
                 crop_size: int = 256, num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        super().__init__()

        analytics = get_analytics(input_path=Path(data_dir),
                                  get_gt_data_paths_func=CroppedRotNet.get_gt_data_paths)

        self.mean = analytics['mean']
        self.std = analytics['std']
        self.class_encodings = np.array(ROTATION_ANGLES)
        self.num_classes = len(self.class_encodings)
        self.class_weights = np.ones(self.num_classes)

        self.image_transform = OnlyImage(transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean=self.mean, std=self.std),
                                                             transforms.RandomCrop(size=crop_size)]))

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_folder_name = data_folder_name
        self.data_dir = validate_path_for_self_supervised(data_dir=data_dir, data_folder_name=self.data_folder_name)

        self.selection_train = selection_train
        self.selection_val = selection_val
        self.selection_test = selection_test

        self.dims = (3, crop_size, crop_size)

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.train = CroppedRotNet(**self._create_dataset_parameters('train'), selection=self.selection_train)
            self.val = CroppedRotNet(**self._create_dataset_parameters('val'), selection=self.selection_val)

            self._check_min_num_samples(num_samples=len(self.train), data_split='train',
                                        drop_last=self.drop_last)
            self._check_min_num_samples(num_samples=len(self.val), data_split='val',
                                        drop_last=self.drop_last)

        if stage == 'test' or stage is not None:
            self.test = CroppedRotNet(**self._create_dataset_parameters('test'), selection=self.selection_test)
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
                'image_transform': self.image_transform,
                'classes': self.class_encodings,
                'is_test': is_test}

