"""
Load a dataset of historic documents by specifying the folder where its located.
"""

from dataclasses import dataclass
# Utils
from pathlib import Path
from typing import List, Tuple, Union, Optional

import torch.utils.data as data
from omegaconf import ListConfig
from torch import is_tensor
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
    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str,
                 image_dims: ImageDimensions,
                 selection: Optional[Union[int, List[str]]] = None,
                 is_test=False, image_transform=None, target_transform=None, twin_transform=None,
                 **kwargs):
        """
        Parameters
        ----------
        path : string
            Path to dataset folder (train / val / test)
        classes :
        workers : int
        imgs_in_memory :
        crops_per_image : int
        crop_size : int
        image_transform : callable
        target_transform : callable
        twin_transform : callable
        loader : callable
            A function to load an image given its path.
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

    def __getitem__(self, index):
        if self.is_test:
            return self._get_test_items(index=index)
        else:
            return self._get_train_val_items(index=index)

    def _get_train_val_items(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        assert img.shape[-2:] == gt.shape[-2:]
        return img, gt

    def _get_test_items(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        assert img.shape[-2:] == gt.shape[-2:]

        return img, gt, index

    def _load_data_and_gt(self, index):
        data_img = pil_loader(self.img_gt_path_list[index][0])
        gt_img = pil_loader(self.img_gt_path_list[index][1])

        assert data_img.height == self.image_dims.height and data_img.width == self.image_dims.width
        assert gt_img.height == self.image_dims.height and gt_img.width == self.image_dims.width

        return data_img, gt_img

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
        if not is_tensor(gt) and gt is not None:
            gt = ToTensor()(gt)

        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt

    @staticmethod
    def get_img_gt_path_list(directory: Path, data_folder_name: str, gt_folder_name: str,
                             selection: Optional[Union[int, List[str]]] = None) \
            -> List[Tuple[Path, Path, str]]:
        """
        Structure of the folder

        directory/data/FILE_NAME.png
        directory/gt/FILE_NAME.png

        :param directory:
        :param data_folder_name:
        :param gt_folder_name:
        :param selection:
        :return: tuple
            (path_data_file, path_gt_file)
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
