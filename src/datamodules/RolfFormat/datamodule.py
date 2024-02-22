from typing import Union, List, Optional, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datamodules.RGB.utils.single_transform import IntegerEncoding
from src.datamodules.RolfFormat.datasets.dataset import DatasetRolfFormat, DatasetSpecs
from src.datamodules.RolfFormat.utils.image_analytics import get_analytics_data, get_analytics_gt
from src.datamodules.base_datamodule import AbstractDatamodule
from src.datamodules.utils.dataset_predict import DatasetPredict
from src.datamodules.utils.misc import ImageDimensions, get_image_dims
from src.datamodules.utils.wrapper_transforms import OnlyImage, OnlyTarget
from src.utils import utils

log = utils.get_logger(__name__)


class DataModuleRolfFormat(AbstractDatamodule):
    """
    DataModule for the RolfFormat dataset. What makes this dataset special is that all the files are within one folder.
    Each file name has a fixed structure of `name_{file_number}.jpg`. The file number is a number between 0 and 9999.
    The different splits are defined by giving a range and a root folder for each split.

    :param data_root: Root folder of the dataset.
    :type data_root: str
    :param train_specs: Dictionary with the specs for the train split.
    :type train_specs: dict
    :param val_specs: Dictionary with the specs for the validation split.
    :param test_specs: Dictionary with the specs for the test split.
    :type val_specs: dict
    :param pred_file_path_list: List of file paths to predict.
    :type pred_file_path_list: List[str]
    :param image_analytics: A dictionary with the mean and std of the images.
    :type image_analytics: dict
    :param classes: A dictionary with the class encodings and weights.
    :type classes: dict
    :param image_dims: The dimensions of the images.
    :type image_dims: ImageDimensions
    :param num_workers: Number of workers for the dataloader.
    :type num_workers: int
    :param batch_size: Batch size for the dataloader.
    :type batch_size: int
    :param shuffle: Whether to shuffle the dataset.
    :type shuffle: bool
    :param drop_last: Whether to drop the last batch if it is smaller than the batch size.
    :type drop_last: bool
    """

    def __init__(self, data_root: str,
                 train_specs: Dict = None, val_specs: Dict = None, test_specs: Dict = None,
                 pred_file_path_list: List[str] = None,
                 image_analytics: Dict = None, classes: Dict = None, image_dims: ImageDimensions = None,
                 num_workers: int = 4, batch_size: int = 8,
                 shuffle: bool = True, drop_last: bool = True):
        """
        Constructor method for the `DataModuleRolfFormat` class.
        """
        super().__init__()

        if train_specs is not None:
            self.train_dataset_specs = [DatasetSpecs(data_root=data_root, **v) for k, v in train_specs.items()]
        if val_specs is not None:
            self.val_dataset_specs = [DatasetSpecs(data_root=data_root, **v) for k, v in val_specs.items()]
        if test_specs is not None:
            self.test_dataset_specs = [DatasetSpecs(data_root=data_root, **v) for k, v in test_specs.items()]
        if pred_file_path_list is not None:
            self.pred_file_path_list = pred_file_path_list

        if image_analytics is None or classes is None or image_dims is None:
            train_paths_data_gt = DatasetRolfFormat.get_img_gt_path_list(list_specs=self.train_dataset_specs)

        if image_dims is None:
            image_dims = get_image_dims(data_gt_path_list=train_paths_data_gt)
            self._print_image_dims(image_dims=image_dims)

        if image_analytics is None:
            analytics_data = get_analytics_data(img_gt_path_list=train_paths_data_gt)
            self._print_analytics_data(analytics_data=analytics_data)
        else:
            analytics_data = {'mean': [image_analytics['mean']['R'],
                                       image_analytics['mean']['G'],
                                       image_analytics['mean']['B']],
                              'std': [image_analytics['std']['R'],
                                      image_analytics['std']['G'],
                                      image_analytics['std']['B']]}

        if classes is None:
            analytics_gt = get_analytics_gt(img_gt_path_list=train_paths_data_gt)
            self._print_analytics_gt(analytics_gt=analytics_gt)
        else:
            analytics_gt = {'class_encodings': [],
                            'class_weights': []}
            for _, class_specs in classes.items():
                analytics_gt['class_encodings'].append([class_specs['color']['R'],
                                                        class_specs['color']['G'],
                                                        class_specs['color']['B']])
                if 'weight' in class_specs:
                    analytics_gt['class_weights'].append(class_specs['weight'])
                else:
                    analytics_gt['class_weights'].append(None)

            if all(x is None for x in analytics_gt['class_weights']):
                analytics_gt['class_weights'] = None
            elif any(x is None for x in analytics_gt['class_weights']):
                log.error('Some classes have a class weight and others do not. '
                          'If you set class weights, you have to do this for all classes.')
                raise ValueError

        self.image_dims = image_dims
        self.dims = (3, self.image_dims.width, self.image_dims.height)

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

        # Check default attributes using base_datamodule function
        self._check_attributes()

    def _print_analytics_data(self, analytics_data):
        indent = 4 * ' '
        lines = ['']
        lines.append('image_analytics:')
        lines.append(f'{indent}mean:')
        lines.append(f'{indent}{indent}R: {analytics_data["mean"][0]}')
        lines.append(f'{indent}{indent}G: {analytics_data["mean"][1]}')
        lines.append(f'{indent}{indent}B: {analytics_data["mean"][2]}')
        lines.append(f'{indent}std:')
        lines.append(f'{indent}{indent}R: {analytics_data["std"][0]}')
        lines.append(f'{indent}{indent}G: {analytics_data["std"][1]}')
        lines.append(f'{indent}{indent}B: {analytics_data["std"][2]}')

        print_string = '\n'.join(lines)
        log.info(print_string)

    def _print_analytics_gt(self, analytics_gt):
        indent = 4 * ' '
        lines = ['']
        lines.append('classes:')
        for i, class_specs in enumerate(zip(analytics_gt['class_encodings'], analytics_gt['class_weights'])):
            lines.append(f'{indent}class{i}:')
            lines.append(f'{indent}{indent}color:')
            lines.append(f'{indent}{indent}{indent}R: {class_specs[0][0]}')
            lines.append(f'{indent}{indent}{indent}G: {class_specs[0][1]}')
            lines.append(f'{indent}{indent}{indent}B: {class_specs[0][2]}')
            lines.append(f'{indent}{indent}weight: {class_specs[1]}')

        print_string = '\n'.join(lines)
        log.info(print_string)

    def _print_image_dims(self, image_dims: ImageDimensions):
        indent = 4 * ' '
        lines = ['']
        lines.append('image_dims:')
        lines.append(f'{indent}width:  {image_dims.width}')
        lines.append(f'{indent}height: {image_dims.height}')

        print_string = '\n'.join(lines)
        log.info(print_string)

    def setup(self, stage: Optional[str] = None):
        super().setup()

        common_kwargs = {'image_dims': self.image_dims,
                         'image_transform': self.image_transform,
                         'target_transform': self.target_transform,
                         'twin_transform': self.twin_transform}

        if stage == 'fit' or stage is None:
            self.train = DatasetRolfFormat(dataset_specs=self.train_dataset_specs,
                                           is_test=False,
                                           **common_kwargs)
            log.info(f'Initialized train dataset with {len(self.train)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.train),
                                       data_split='train', drop_last=self.drop_last)

            self.val = DatasetRolfFormat(dataset_specs=self.val_dataset_specs,
                                         is_test=False,
                                         **common_kwargs)
            log.info(f'Initialized val dataset with {len(self.val)} samples.')
            self.check_min_num_samples(self.trainer.num_devices, self.batch_size, num_samples=len(self.val),
                                       data_split='val', drop_last=self.drop_last)

        if stage == 'test':
            self.test = DatasetRolfFormat(dataset_specs=self.test_dataset_specs,
                                          is_test=True,
                                          **common_kwargs)
            log.info(f'Initialized test dataset with {len(self.test)} samples.')
            # self._check_min_num_samples(num_samples=len(self.test), data_split='test', drop_last=False)

        if stage == 'predict':
            self.predict = DatasetPredict(image_path_list=self.pred_file_path_list,
                                          **common_kwargs)
            log.info(f'Initialized predict dataset with {len(self.predict)} samples.')
            # self._check_min_num_samples(num_samples=len(self.test), data_split='test', drop_last=False)

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

        :param index: Index of the sample we want the filename of.
        :type index: int
        :raises Exception: This method can just be called during testing
        :return: Filename of the doc image.
        :rtype: str
        """
        if not hasattr(self, 'test'):
            raise ValueError('This method can just be called during testing')

        return self.test.output_file_list[index]

    def get_output_filename_predict(self, index: int) -> str:
        """
        Returns the original filename of the doc image.
        You can just use this during testing!

        :param index: Index of the sample we want the filename of.
        :type index: int
        :raises Exception: This method can just be called during prediction
        :return: Filename of the doc image.
        :rtype: str
        """
        if not hasattr(self, 'predict'):
            raise ValueError('This method can just be called during prediction')

        return self.predict.output_file_list[index]
