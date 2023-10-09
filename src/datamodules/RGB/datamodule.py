from pathlib import Path
from typing import Union, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.RGB.datasets.full_page_dataset import DatasetRGB
from src.datamodules.RGB.utils.image_analytics import get_analytics
from src.datamodules.RGB.utils.single_transform import IntegerEncoding
from src.datamodules.base_datamodule import AbstractDatamodule
from src.datamodules.utils.dataset_predict import DatasetPredict
from src.datamodules.utils.misc import validate_path_for_segmentation, ImageDimensions
from src.datamodules.utils.wrapper_transforms import OnlyImage, OnlyTarget
from src.utils import utils

log = utils.get_logger(__name__)


class DataModuleRGB(AbstractDatamodule):
    """
    The data module for a dataset where the classes of the ground truth are encoded as colors in the image.
    This data module expects full sized images. This does not mean, that the image needs to be in the original resolution
    but it can not consist of cropped images. If you want to work with cropped images use class: `DataModuleCroppedRGB`.

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
    :param pred_file_path_list: List of file paths to the images that should be predicted.
    :type pred_file_path_list: List[str]
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
                 pred_file_path_list: List[str] = None,
                 selection_train: Optional[Union[int, List[str]]] = None,
                 selection_val: Optional[Union[int, List[str]]] = None,
                 selection_test: Optional[Union[int, List[str]]] = None,
                 num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        """
        Constructor of the class: `DataModuleRGB`.
        """
        super().__init__()

        self.train_folder_name = train_folder_name
        self.val_folder_name = val_folder_name
        self.test_folder_name = test_folder_name
        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name

        if pred_file_path_list is not None:
            self.pred_file_path_list = pred_file_path_list

        analytics_data, analytics_gt = get_analytics(input_path=Path(data_dir),
                                                     data_folder_name=self.data_folder_name,
                                                     gt_folder_name=self.gt_folder_name,
                                                     train_folder_name=self.train_folder_name,
                                                     get_img_gt_path_list_func=DatasetRGB.get_img_gt_path_list)

        self.image_dims = ImageDimensions(width=analytics_data['width'], height=analytics_data['height'])
        self.dims = (3, self.image_dims.height, self.image_dims.width)

        self.mean = analytics_data['mean']
        self.std = analytics_data['std']
        self.class_encodings = analytics_gt['class_encodings']
        self.class_encodings_tensor = torch.tensor(self.class_encodings) / 255
        self.num_classes = len(self.class_encodings)
        self.class_weights = torch.as_tensor(analytics_gt['class_weights'])

        self.twin_transform = None
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

        # Check default attributes using base_datamodule function
        self._check_attributes()

    def setup(self, stage: Optional[str] = None):
        super().setup()

        common_kwargs = {'image_dims': self.image_dims,
                         'image_transform': self.image_transform,
                         'target_transform': self.target_transform,
                         'twin_transform': self.twin_transform}

        dataset_kwargs = {'data_folder_name': self.data_folder_name,
                          'gt_folder_name': self.gt_folder_name}

        if stage == 'fit' or stage is None:
            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.train_folder_name)
            self.train = DatasetRGB(path=self.data_dir / self.train_folder_name,
                                    selection=self.selection_train,
                                    is_test=False,
                                    **dataset_kwargs,
                                    **common_kwargs)
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.train),
                                       data_split=self.train_folder_name,
                                       drop_last=self.drop_last)

            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.val_folder_name)
            self.val = DatasetRGB(path=self.data_dir / self.val_folder_name,
                                  selection=self.selection_val,
                                  is_test=False,
                                  **dataset_kwargs,
                                  **common_kwargs)
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.val),
                                       data_split=self.val_folder_name,
                                       drop_last=self.drop_last)

        if stage == 'test':
            self.data_dir = validate_path_for_segmentation(data_dir=self.data_dir,
                                                           data_folder_name=self.data_folder_name,
                                                           gt_folder_name=self.gt_folder_name,
                                                           split_name=self.test_folder_name)
            self.test = DatasetRGB(path=self.data_dir / self.test_folder_name,
                                   selection=self.selection_test,
                                   is_test=True,
                                   **dataset_kwargs,
                                   **common_kwargs)
            log.info(f'Initialized test dataset with {len(self.test)} samples.')

        if stage == 'predict':
            self.predict = DatasetPredict(image_path_list=self.pred_file_path_list,
                                          **common_kwargs)
            log.info(f'Initialized predict dataset with {len(self.predict)} samples.')

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

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.predict,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

    def get_output_filename_test(self, index: int) -> str:
        """
        Returns the original filename of the doc image.
        You can just use this during testing!

        :param index: index of the sample
        :type index: int
        :return: original filename of the doc image
        :rtype: str
        """
        if not hasattr(self, 'test'):
            raise Exception('This method can just be called during testing')

        return self.test.output_file_list[index]

    def get_output_filename_predict(self, index: int) -> str:
        """
        Returns the original filename of the doc image.
        You can just use this during testing!

        :param index: index of the sample
        :type index: int
        :return: original filename of the doc image
        :rtype: str
        """
        if not hasattr(self, 'predict'):
            raise Exception('This method can just be called during prediction')

        return self.predict.output_file_list[index]
