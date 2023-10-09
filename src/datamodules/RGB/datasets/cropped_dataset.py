"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any

import torch.utils.data as data
from PIL import Image
from omegaconf import ListConfig
from torch import is_tensor, Tensor
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension
from torchvision.transforms import ToTensor

from src.datamodules.utils.misc import selection_validation
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.gif')

log = utils.get_logger(__name__)


class CroppedDatasetRGB(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png

        :param path: Path to dataset folder (train / val / test)
        :type path: Path
        :param data_folder_name: name of the folder that contains the data
        :type data_folder_name: str
        :param gt_folder_name: name of the folder that contains the ground truth
        :type gt_folder_name: str
        :param selection: selection of the data, defaults to None
        :type selection: Optional[Union[int, List[str]]], optional
        :param is_test: flag to indicate if the dataset is used for testing, defaults to False
        :type is_test: bool, optional
        :param image_transform: image transformation, defaults to None
        :type image_transform: callable, optional
        :param target_transform: target transformation, defaults to None
        :type target_transform: callable, optional
        :param twin_transform: twin transformation, defaults to None
        :type twin_transform: callable, optional

    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str,
                 selection: Optional[Union[int, List[str]]] = None,
                 is_test: bool = False, image_transform: callable = None, target_transform: callable = None,
                 twin_transform: callable = None):
        """
        Constructor method for the class :class: `CroppedDatasetRGB`.
        """

        self.path = path
        self.data_folder_name = data_folder_name
        self.gt_folder_name = gt_folder_name
        self.selection = selection

        # transformations
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.twin_transform = twin_transform

        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_paths_per_page = self.get_gt_data_paths(path, data_folder_name=self.data_folder_name,
                                                         gt_folder_name=self.gt_folder_name, selection=self.selection)

        self.num_samples = len(self.img_paths_per_page)
        if self.num_samples == 0:
            raise RuntimeError("Found 0 images in subfolders of: {} \n Supported image extensions are: {}".format(
                path, ",".join(IMG_EXTENSIONS)))

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor]]:
        if self.is_test:
            return self._get_test_items(index=index)
        else:
            return self._get_train_val_items(index=index)

    def _get_train_val_items(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Returns the image and the ground truth image at the given index. If transformations have been defined,
        they are applied here.
        :param index: index of the image to return
        :type index: int
        :return: The image and the corresponding ground truth image with transformations applied
        :rtype: Tuple[Tensor, Tensor]
        """
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        return img, gt

    def _get_test_items(self, index: int) -> Tuple[Tensor, Tensor, int]:
        """
        Returns the image and the ground truth image at the given index for testing. If transformations have been defined,
        they are applied here. Additionally, to the :method: `_get_train_val_items`, the index of the image is returned.
        :param index: index of the image to return
        :type index: int
        :return: The image and the corresponding ground truth image with transformations applied
        :rtype: Tuple[Tensor, Tensor, int]
        """
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        return img, gt, index

    def _load_data_and_gt(self, index: int) -> Tuple[Image.Image, Image.Image]:
        """
        Loads the image and the ground truth image at the given index.
        :param index: index of the image to return
        :type index: int
        :return: The image and the corresponding ground truth image
        :rtype: Tuple[Image.Image, Image.Image]
        """
        data_img = pil_loader(self.img_paths_per_page[index][0])
        gt_img = pil_loader(self.img_paths_per_page[index][1])

        return data_img, gt_img

    def _apply_transformation(self, img: Union[Image.Image, Tensor], gt: Union[Image.Image, Tensor]) \
            -> Tuple[Tensor, Tensor]:
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        :param img: The original image to apply the transformations to
        :type img: Union[Image.Image, Tensor]
        :param gt: The corresponding ground truth image to apply the transformations to
        :type img: Union[Image.Image, Tensor]
        :return: The transformed image and the transformed ground truth image
        :rtype: Tuple[Tensor, Tensor]
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

        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt

    @staticmethod
    def get_gt_data_paths(directory: Path, data_folder_name: str, gt_folder_name: str,
                          selection: Optional[Union[int, List[str]]] = None) \
            -> List[Tuple[Any, Any, str, Any]]:
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

        # get all subitems (and files) sorted
        subitems = sorted(path_data_root.iterdir())

        # check the selection parameter
        if selection:
            selection = selection_validation(subitems, selection, full_page=False)

        counter = 0  # Counter for subdirectories, needed for selection parameter

        for path_data_subdir in subitems:
            if not path_data_subdir.is_dir():
                if has_file_allowed_extension(path_data_subdir.name, IMG_EXTENSIONS):
                    log.warning("image file found in data root: " + str(path_data_subdir))
                continue

            counter += 1

            if selection:
                if isinstance(selection, int):
                    if counter > selection:
                        break

                elif isinstance(selection, ListConfig) or isinstance(selection, list):
                    if path_data_subdir.name not in selection:
                        continue

            path_gt_subdir = path_gt_root / path_data_subdir.stem
            assert path_gt_subdir.is_dir()

            for path_data_file, path_gt_file in zip(sorted(path_data_subdir.iterdir()),
                                                    sorted(path_gt_subdir.iterdir())):
                assert has_file_allowed_extension(path_data_file.name, IMG_EXTENSIONS) == \
                       has_file_allowed_extension(path_gt_file.name, IMG_EXTENSIONS), \
                    'get_img_gt_path_list(): image file aligned with non-image file'

                if has_file_allowed_extension(path_data_file.name, IMG_EXTENSIONS) and \
                        has_file_allowed_extension(path_gt_file.name, IMG_EXTENSIONS):
                    assert path_data_file.stem == path_gt_file.stem, \
                        'get_img_gt_path_list(): mismatch between data filename and gt filename'
                    paths.append((path_data_file, path_gt_file, path_data_subdir.stem, path_data_file.stem))

        return paths
