import random

from torchvision.transforms import functional as F


class TwinCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class TwinRandomCrop(object):
    """Crop the given PIL Images at the same random location"""

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def get_params(self, img_size):
        """
        Get parameters for ``crop`` for a random crop

        :param img_size: (tuple) Image size (h, w)
        :returns: (tuple) params (i, j, h, w) to be passed to ``crop`` for random crop.
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
    """

    def __call__(self, img, gt):
        """
        :param img: (PIL Image or numpy.ndarray) Image to be converted to tensor.
        :param gt: (PIL Image or numpy.ndarray) Image to be converted to tensor.
        :returns: (Tensor, Tensor) Converted image.
        """
        return F.to_tensor(img), F.to_tensor(gt)


class ToTensorSlidingWindowCrop(object):
    """
    Crop the data and ground truth image at the specified coordinates to the specified size and convert
    them to a tensor.
    """

    def __init__(self, crop_size):
        """
        :param crop_size: (int) Size of the crop.
        """
        self.crop_size = crop_size

    def __call__(self, img, gt, coordinates):
        """
        :param img: (PIL Image) Data image to be cropped and converted to tensor.
        :param gt: (PIL Image) Ground truth image to be cropped and converted to tensor.
        :param coordinates: (tuple) Coordinates of the top left corner of the crop.
        :returns: (Tensor, Tensor) Converted image.
        """
        x_position = coordinates[0]
        y_position = coordinates[1]

        return F.to_tensor(F.crop(img, x_position, y_position, self.crop_size, self.crop_size)), \
               F.to_tensor(F.crop(gt, x_position, y_position, self.crop_size, self.crop_size))
