"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional

from torch import is_tensor
from torchvision.transforms import ToTensor

from src.datamodules.RGB.datasets.cropped_dataset import CroppedDatasetRGB
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm')

log = utils.get_logger(__name__)


class CroppedHisDBDataset(CroppedDatasetRGB):
    """Dataset used for the `DivaHisDB dataset<https://ieeexplore.ieee.org/abstract/document/7814109>`_ in a cropped setup. This class represents one split of the whole dataset.

    The structure of the folder should be as follows::

        path
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

    :param path: Path to the dataset
    :type path: Path
    :param data_folder_name: name of the folder that contains the original images
    :type data_folder_name: str
    :param gt_folder_name: name of the folder that contains the ground truth images
    :type gt_folder_name: str
    :param selection: filtering of the dataset, can be an integer or a list of strings
    :type selection: Union[int, List[str], None]
    :param is_test: if True, :meth:`__getitem__` will return the index of the image
    :type is_test: bool
    :param image_transform: transformation that is applied to the image
    :type image_transform: callable
    :param target_transform: transformation that is applied to the target
    :type target_transform: callable
    :param twin_transform: transformation that is applied to both image and target
    :type twin_transform: callable
    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str,
                 selection: Optional[Union[int, List[str]]] = None,
                 is_test=False, image_transform=None, target_transform=None, twin_transform=None,
                 **kwargs):
        """
        Constructor method for the CroppedHisDBDataset class.
        """
        super().__init__(path, data_folder_name, gt_folder_name, selection, is_test, image_transform, target_transform,
                         twin_transform, **kwargs)

    def _get_train_val_items(self, index):
        data_img, gt_img = super()._load_data_and_gt(index=index)
        img, gt, boundary_mask = self._apply_transformation(data_img, gt_img)
        return img, gt, boundary_mask

    def _get_test_items(self, index):
        data_img, gt_img = super()._load_data_and_gt(index=index)
        img, gt, boundary_mask = self._apply_transformation(data_img, gt_img)
        return img, gt, boundary_mask, index

    def _apply_transformation(self, img, gt):
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        :param img: Original image to apply transformation on
        :type img: PIL image
        :param gt: Ground truth image to apply transformation on
        :type gt: PIL image
        :return: transformed image and gt
        :rtype: tuple
        """
        if self.twin_transform is not None and not self.is_test:
            img, gt = self.twin_transform(img, gt)

        if self.image_transform is not None:
            # perform transformations
            img, gt = self.image_transform(img, gt)

        if not is_tensor(img):
            img = ToTensor()(img)
        if not is_tensor(gt):
            gt = ToTensor()(gt)

        border_mask = gt[0, :, :] != 0
        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt, border_mask
