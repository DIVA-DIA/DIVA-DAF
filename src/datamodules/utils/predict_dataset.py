from typing import List

import torch.utils.data as data
from torch import is_tensor
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor

from src.utils import utils

log = utils.get_logger(__name__)
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif']


class PredictDataset(data.Dataset):

    def __init__(self, image_path_list: List[str],
                 image_transform=None, target_transform=None, twin_transform=None,
                 classes=None, **kwargs):
        """
        Parameters
        ----------
        classes :
        image_transform : callable
        target_transform : callable
        twin_transform : callable
        """

        self.image_path_list = image_path_list

        # Init list
        self.classes = classes
        # self.crops_per_image = crops_per_image

        # transformations
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.num_samples = len(self.image_path_list)
        if self.num_samples == 0:
            raise RuntimeError(f'List of image paths is empty!')

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        return self.num_samples

    def __getitem__(self, index):
        data_img = self._load_data_and_gt(index=index)
        data_tensor = self._apply_transformation(img=data_img)
        return data_tensor

    def _load_data_and_gt(self, index):
        data_img = pil_loader(self.image_path_list[index])
        return data_img

    def _apply_transformation(self, img):
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        Parameters
        ----------
        img: PIL image
            image data
        Returns
        -------
        tuple
            img and gt after transformations
        """
        if self.image_transform is not None:
            # perform transformations
            img, _ = self.image_transform(img, None)

        if not is_tensor(img):
            img = ToTensor()(img)

        return img
