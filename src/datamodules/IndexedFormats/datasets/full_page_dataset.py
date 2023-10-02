"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from omegaconf import ListConfig
from torch import is_tensor
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension
from torchvision.transforms import ToTensor

from src.datamodules.utils.misc import ImageDimensions, selection_validation, pil_loader_gif, get_output_file_list
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif')
GT_EXTENSION = ('.gif')

log = utils.get_logger(__name__)


class DatasetIndexed(data.Dataset):
    """A dataset where the images are arranged in this way:

        root/gt/xxx.gif
        root/gt/xxy.gif
        root/gt/xxz.gif

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png

        And the ground truth is represented in an index format like GIF.

        :param path: Path to the dataset
        :type path: Path
        :param data_folder_name: Name of the folder where the data is located
        :type data_folder_name: str
        :param gt_folder_name: Name of the folder where the ground truth is located
        :type gt_folder_name: str
        :param image_dims: Image dimensions of the dataset
        :type image_dims: ImageDimensions
        :param is_test: Flag to indicate if the dataset is used for testing
        :type is_test: bool
        :param selection: Selection of the dataset, can be an integer or a list of strings
        :type selection: Optional[Union[int, List[str]]]
        :param image_transform: Transformations that are applied to the image
        :type image_transform: Optional[Callable]
    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str,
                 image_dims: ImageDimensions, is_test=False,
                 selection: Optional[Union[int, List[str]]] = None,
                 image_transform=None) -> None:
        """
         Constructor method for the DatasetIndexed class.
        """

        self.path = path
        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name
        self.selection = selection

        self.image_dims = image_dims

        # transformations
        self.image_transform = image_transform

        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_gt_path_list = self.get_img_gt_path_list(path, data_folder_name=self.data_folder_name,
                                                          gt_folder_name=self.gt_folder_name, selection=self.selection)
        if is_test:
            self.image_path_list = [img_gt_path[0] for img_gt_path in self.img_gt_path_list]
            self.output_file_list = get_output_file_list(image_path_list=self.image_path_list)

        self.num_samples = len(self.img_gt_path_list)
        if self.num_samples == 0:
            raise RuntimeError(f"Found 0 images in: {path} \n "
                               f"Supported image extensions are: {' '.join(IMG_EXTENSIONS)}\n"
                               f"Supported ground truth extensions are: {' '.join(GT_EXTENSION)}")

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        assert img.shape[-2:] == gt.shape[-2:]
        if self.is_test:
            return img, gt, index
        else:
            return img, gt

    def _load_data_and_gt(self, index: int) -> Tuple[Image.Image, Image.Image]:
        data_img = pil_loader(str(self.img_gt_path_list[index][0]))
        gt_img = pil_loader_gif(self.img_gt_path_list[index][1])

        assert data_img.height == self.image_dims.height and data_img.width == self.image_dims.width
        assert gt_img.height == self.image_dims.height and gt_img.width == self.image_dims.width

        return data_img, gt_img

    def _apply_transformation(self, img: Image, gt: Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformations to the image and the ground truth.
        :param img: Original image
        :type img: Image
        :param gt: Ground truth as an image
        :type gt: Image
        :return: Original and ground Truth as Tensor with applied transformations
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.image_transform is not None:
            # perform transformations
            img, _ = self.image_transform(img, gt)

        if not is_tensor(img):
            img = ToTensor()(img)

        # remove first dim s.t. gt is just w x h
        gt_np = np.asarray(gt)
        if len(gt_np.shape) == 3:
            gt_np = np.squeeze(gt_np, axis=0)
        gt = torch.tensor(gt_np, dtype=torch.long)

        return img, gt

    @staticmethod
    def get_img_gt_path_list(directory: Path, data_folder_name: str, gt_folder_name: str,
                             selection: Optional[Union[int, List[str]]] = None) \
            -> List[Tuple[Path, Path]]:
        """
        Structure of the folder

        directory/data/FILE_NAME.png
        directory/gt/FILE_NAME.gif

        :param directory: Path to the dataset
        :type directory: Path
        :param data_folder_name: Name of the folder where the data is located
        :type data_folder_name: str
        :param gt_folder_name: Name of the folder where the ground truth is located
        :type gt_folder_name: str
        :param selection: Selection of the dataset, can be an integer or a list of strings
        :type selection: Optional[Union[int, List[str]]]
        :return: List of tuples with the path to the data and the ground truth
        :rtype: List[Tuple[Path, Path]]
        """
        paths = []
        directory = directory.expanduser()

        path_data_root = directory / data_folder_name
        path_gt_root = directory / gt_folder_name

        if not (path_data_root.is_dir() or path_gt_root.is_dir()):
            log.error("folder data or gt not found in " + str(directory))

        # get all files sorted
        files_in_data_root = sorted(path_data_root.iterdir())

        # check the selection parameter
        if selection:
            selection = selection_validation(files_in_data_root, selection, full_page=True)

        counter = 0  # Counter for subdirectories, needed for selection parameter

        for path_data_file, path_gt_file in zip(sorted(files_in_data_root), sorted(path_gt_root.iterdir())):
            counter += 1

            if selection:
                if isinstance(selection, int):
                    if counter > selection:
                        break

                elif isinstance(selection, ListConfig) or isinstance(selection, list):
                    if path_data_file.stem not in selection:
                        continue

            assert has_file_allowed_extension(path_data_file.name, IMG_EXTENSIONS) == \
                   has_file_allowed_extension(path_gt_file.name, GT_EXTENSION), \
                'get_img_gt_path_list(): image file aligned with non-image file'

            paths.append((path_data_file, path_gt_file))

        return paths
