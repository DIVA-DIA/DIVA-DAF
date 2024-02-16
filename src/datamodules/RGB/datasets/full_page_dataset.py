"""
Load a dataset of historic documents by specifying the folder where its located.
"""

from dataclasses import dataclass
# Utils
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any

import torch.utils.data as data
from omegaconf import ListConfig
from PIL import Image
from torch import is_tensor, Tensor
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension
from torchvision.transforms import ToTensor

from src.datamodules.utils.misc import ImageDimensions, get_output_file_list, selection_validation
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif')

log = utils.get_logger(__name__)


class DatasetRGB(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png

        :param path: path to the dataset
        :type path: Path
        :param data_folder_name: name of the folder where the data is located
        :type data_folder_name: str
        :param gt_folder_name: name of the folder where the ground truth is located
        :type gt_folder_name: str
        :param image_dims: dimensions of the image
        :type image_dims: ImageDimensions
        :param selection: selection of the data, can be an int or a list of strings
        :type selection: Optional[Union[int, List[str]]]
        :param is_test: flag to indicate if the dataset is used for testing
        :type is_test: bool, optional
        :param image_transform: image transformation
        :type image_transform: callable, optional
        :param target_transform: target transformation
        :type target_transform: callable, optional
        :param twin_transform: twin transformation
        :type twin_transform: callable, optional
    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str,
                 image_dims: ImageDimensions,
                 selection: Optional[Union[int, List[str]]] = None,
                 is_test: bool = False, image_transform: callable = None, target_transform: callable = None,
                 twin_transform: callable = None,
                 **kwargs):
        """


        """

        self.path = path
        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name
        self.selection = selection

        self.image_dims = image_dims

        # transformations
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_gt_path_list = self.get_img_gt_path_list(path, data_folder_name=self.data_folder_name,
                                                          gt_folder_name=self.gt_folder_name, selection=self.selection)

        if is_test:
            self.image_path_list = [img_gt_path[0] for img_gt_path in self.img_gt_path_list]
            self.output_file_list = get_output_file_list(image_path_list=self.image_path_list)

        self.num_samples = len(self.img_gt_path_list)
        if self.num_samples == 0:
            raise RuntimeError("Found 0 images in: {} \n Supported image extensions are: {}".format(
                path, ",".join(IMG_EXTENSIONS)))

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, str]]:
        """
        This function returns the data and the ground truth for a given index. If the dataset is used for testing,
        the index is used to get the image and the ground truth. If the dataset is used for training or validation,
        the index is used to get the coordinates where the sliding window should be cropped.

        :param index: index of the image
        :type index: int
        :return: the item at the given index
        :rtype: Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, str]]
        """
        if self.is_test:
            return self._get_test_items(index=index)
        else:
            return self._get_train_val_items(index=index)

    def _get_train_val_items(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        This function returns the data and the ground truth for a given index.

        :param index: index of the image
        :type index: int
        :return: the item at the given index
        :rtype: Tuple[Tensor, Tensor]
        """

        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        assert img.shape[-2:] == gt.shape[-2:]
        return img, gt

    def _get_test_items(self, index: int) -> Tuple[Tensor, Tensor, int]:
        """
        This function returns the data and the ground truth for a given index. Additionally, the index is returned.

        :param index: index of the image
        :type index: int
        :return: the item at the given index
        :rtype: Tuple[Tensor, Tensor, str]
        """
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        assert img.shape[-2:] == gt.shape[-2:]

        return img, gt, index

    def _load_data_and_gt(self, index: int) -> Tuple[Image.Image, Image.Image]:
        """
        This function loads the data and the ground truth for a given index.

        :param index: index of the image
        :type index: int
        :return: the item at the given index
        :rtype: Tuple[Image.Image, Image.Image]
        """
        data_img = pil_loader(self.img_gt_path_list[index][0])
        gt_img = pil_loader(self.img_gt_path_list[index][1])

        assert data_img.height == self.image_dims.height and data_img.width == self.image_dims.width
        assert gt_img.height == self.image_dims.height and gt_img.width == self.image_dims.width

        return data_img, gt_img

    def _apply_transformation(self, img: Union[Tensor, Image.Image], gt: Union[Tensor, Image.Image]) \
            -> Tuple[Tensor, Tensor]:
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        :param img: Original image
        :type img: Union[Tensor, Image.Image]
        :param gt: Corresponding ground truth image
        :type gt: Union[Tensor, Image.Image]
        :return: Transformed image and ground truth
        :rtype: Tuple[Tensor, Tensor]
        """
        if self.twin_transform is not None and not self.is_test:
            img, gt = self.twin_transform(img, gt)

        if self.image_transform is not None:
            # perform transformations
            img, gt = self.image_transform(img, gt)

        if not is_tensor(img):
            img = ToTensor()(img)
        if not is_tensor(gt) and gt is not None:
            gt = ToTensor()(gt)

        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt

    @staticmethod
    def get_img_gt_path_list(directory: Path, data_folder_name: str, gt_folder_name: str,
                             selection: Optional[Union[int, List[str]]] = None) \
            -> List[Tuple[Any, Any, Any]]:
        """
        Returns a list of tuples that contain the path to the gt and image that belong together.

        Structure of the folder

        directory/data/ORIGINAL_FILENAME/FILE_NAME_X_Y.png
        directory/gt/ORIGINAL_FILENAME/FILE_NAME_X_Y.png

        :param directory: Path to dataset folder (train / val / test)
        :type directory: Path
        :param data_folder_name: name of the folder that contains the data
        :type data_folder_name: str
        :param gt_folder_name: name of the folder that contains the ground truth
        :type gt_folder_name: str
        :param selection: selection of the data, defaults to None
        :type selection: Optional[Union[int, List[str]]], optional
        :return: List of tuples that contain the path to the gt and image that belong together
        :rtype: List[Tuple[Any, Any, str, Any]]
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
                   has_file_allowed_extension(path_gt_file.name, IMG_EXTENSIONS), \
                'get_img_gt_path_list(): image file aligned with non-image file'

            paths.append((path_data_file, path_gt_file, path_data_file.stem))

        return paths
