from glob import glob
from pathlib import Path
from typing import List, Tuple

import torch.utils.data as data
from torch import is_tensor, Tensor
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor
from PIL import Image

from src.datamodules.utils.misc import ImageDimensions, get_output_file_list
from src.utils import utils

log = utils.get_logger(__name__)


class DatasetPredict(data.Dataset):
    """
    Dataset class for the prediction of the test set. It takes a folder of images and creates the prediction of these images.

    :param image_path_list: list of image paths
    :type image_path_list: List[str]
    :param image_dims: image dimensions
    :type image_dims: ImageDimensions
    :param image_transform: image transformation
    :type image_transform: Callable
    :param target_transform: target transformation
    :type target_transform: Callable
    :param twin_transform: twin transformation
    :type twin_transform: Callable
    """

    def __init__(self, image_path_list: List[str], image_dims: ImageDimensions,
                 image_transform=None, target_transform=None, twin_transform=None):
        """
        Constructor method for the DatasetPredict class.
        """

        self._raw_image_path_list = list(image_path_list)
        self.image_path_list = self.expend_glob_path_list(glob_path_list=self._raw_image_path_list)
        self.output_file_list = get_output_file_list(image_path_list=self.image_path_list)

        self.image_dims = image_dims

        # transformations
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.num_samples = len(self.image_path_list)
        if self.num_samples == 0:
            raise RuntimeError(f'List of image paths is empty!')

    def __len__(self) -> int:
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.

        :returns: number of samples
        :rtype: int
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        This function returns the data and the gt for a given index.

        :param index: index of the sample
        :type index: int
        :return: data and gt
        :rtype: Tuple[Tensor, int]
        """

        data_img = self._load_data_and_gt(index=index)
        data_tensor = self._apply_transformation(img=data_img)
        return data_tensor, index

    def _load_data_and_gt(self, index: int) -> Image:
        """
        Loads the data and gt from the disk.

        :param index: index of the sample
        :type index: int
        :returns: The image at the given index
        :rtype: Image
        """
        data_img = pil_loader(self.image_path_list[index])

        assert data_img.height == self.image_dims.height and data_img.width == self.image_dims.width

        return data_img

    def _apply_transformation(self, img: Image) -> Tensor:
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        :param img: Original image to apply transformation on
        :type img: PIL image
        :returns: transformed image
        :rtype: Tensor
        """
        if self.image_transform is not None:
            # perform transformations
            img, _ = self.image_transform(img, None)

        if not is_tensor(img):
            img = ToTensor()(img)

        return img

    @staticmethod
    def expend_glob_path_list(glob_path_list: List[str]) -> List[Path]:
        """
        Expends the glob path list to a list of paths.

        :param glob_path_list: list of glob paths
        :type glob_path_list: List[str]
        :returns: list of paths
        :rtype: List[Path]
        """
        output_list = []
        for glob_path in glob_path_list:
            for s in sorted(glob(glob_path)):
                path = Path(s)
                if path not in output_list:
                    output_list.append(Path(s))
        return output_list
