from pathlib import Path
from typing import Union, List, Optional

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.datamodules.Classification.utils.image_analytics import get_analytics_data_image_folder
from src.datamodules.Classification.utils.misc import validate_path_for_classification
from src.datamodules.base_datamodule import AbstractDatamodule
from src.datamodules.utils.misc import get_image_dims
from src.utils import utils

log = utils.get_logger(__name__)


class ClassificationDatamodule(AbstractDatamodule):
    def __init__(self, data_dir: str,
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        """

        :param data_dir:
        :param selection_train:
        :param selection_val:
        :param num_workers:
        :param batch_size:
        :param shuffle:
        :param drop_last:
        """
        super().__init__()

        analytics_data = get_analytics_data_image_folder(input_path=Path(data_dir))

        self.mean = analytics_data['mean']
        self.std = analytics_data['std']
        # error

        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=self.mean, std=self.std),
                                                   ])

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_dir = validate_path_for_classification(data_dir=data_dir)

        self.selection_train = selection_train
        self.selection_val = selection_val

        train_set = ImageFolder(**self._create_dataset_parameters('train'))
        self.classes = train_set.classes
        self.num_classes = len(self.classes)

        image_dims = get_image_dims(
            data_gt_path_list=train_set.imgs)
        self.image_dims = image_dims
        self.dims = (3, self.image_dims.width, self.image_dims.height)

        self.train = None
        self.val = None

        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: Optional[str] = None):
        super().setup()
        if stage == 'fit' or stage is None:
            self.train = ImageFolder(**self._create_dataset_parameters('train'))
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.train),
                                       data_split='train',
                                       drop_last=self.drop_last)

            self.val = ImageFolder(**self._create_dataset_parameters('val'))
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.val),
                                       data_split='val',
                                       drop_last=self.drop_last)

        if stage == 'test':
            raise ValueError('Test data is not available for Classification.')

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
        raise ValueError('Test data is not available for Classification.')

    def _create_dataset_parameters(self, dataset_type: str = 'train'):
        return {'root': self.data_dir / dataset_type,
                'transform': self.image_transform,
                }
