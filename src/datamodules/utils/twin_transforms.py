import random
from typing import List, Callable, Union, Tuple

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F


class TwinCompose(object):
    """
    Composes several transforms together and applies it to both, the codex image and the ground truth.

    :param transforms: List of transforms to compose.

    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: Union[Tensor, np.ndarray], gt: Union[Tensor, np.ndarray]):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class TwinRandomCrop(object):
    """
    Crop the given PIL Images at the same random location

    :param crop_size: Desired output size of the crop.
    :type crop_size: int
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def get_params(self, img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get parameters for ``crop`` for a random crop

        :param img_size: Image size (h, w)
        :type img_size: Tuple[int, int]
        :returns: params (i, j, h, w) to be passed to ``crop`` for random crop.
        :rtype: Tuple[int, int, int, int]
        """
        w, h = img_size
        th = self.crop_size
        tw = self.crop_size

        assert w >= tw and h >= th

        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, gt):
        i, j, h, w = self.get_params(img.size)
        return F.crop(img, i, j, h, w), F.crop(gt, i, j, h, w)


class TwinImageToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (W x H x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    :param img: Image to be converted to tensor.
    :type img: PIL Image or numpy.ndarray
    :param gt: Image to be converted to tensor.
    :type gt: PIL Image or numpy.ndarray
    :returns: Converted image.
    :rtype: Tuple[Tensor, Tensor]
    """

    def __call__(self, img, gt):
        return F.to_tensor(img), F.to_tensor(gt)


class ToTensorSlidingWindowCrop(object):
    """
    Crop the data and ground truth image at the specified coordinates to the specified size and convert
    them to a tensor.

    :param crop_size: Size of the crop.
    :type crop_size: int
    """

    def __init__(self, crop_size: int):
        """
        Constructor method for the ToTensorSlidingWindowCrop class.
        """
        self.crop_size = crop_size

    def __call__(self, img: Image, gt: Image, coordinates: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        """
        """
        x_position = coordinates[0]
        y_position = coordinates[1]

        return F.to_tensor(F.crop(img, x_position, y_position, self.crop_size, self.crop_size)), \
            F.to_tensor(F.crop(gt, x_position, y_position, self.crop_size, self.crop_size))
