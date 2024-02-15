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
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png
    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str,
                 selection: Optional[Union[int, List[str]]] = None,
                 is_test=False, image_transform=None, target_transform=None, twin_transform=None,
                 **kwargs):

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

        Parameters
        ----------
        img: PIL image
            image data
        gt: PIL image
            ground truth image
        coordinates: tuple (int, int)
            coordinates where the sliding window should be cropped
        Returns
        -------
        tuple
            img and gt after transformations
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
