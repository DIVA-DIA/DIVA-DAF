"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional

import torch.utils.data as data
from omegaconf import ListConfig
from torch import is_tensor
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension
from torchvision.transforms import ToTensor

from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm')

log = utils.get_logger(__name__)


class CroppedHisDBDataset(data.Dataset):
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

    def __getitem__(self, index):
        if self.is_test:
            return self._get_test_items(index=index)
        else:
            return self._get_train_val_items(index=index)

    def _get_train_val_items(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt, boundary_mask = self._apply_transformation(data_img, gt_img)
        return img, gt, boundary_mask

    def _get_test_items(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt, boundary_mask = self._apply_transformation(data_img, gt_img)
        return img, gt, boundary_mask, index

    def _load_data_and_gt(self, index):
        data_img = pil_loader(self.img_paths_per_page[index][0])
        gt_img = pil_loader(self.img_paths_per_page[index][1])

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
        if not is_tensor(gt):
            gt = ToTensor()(gt)

        border_mask = gt[0, :, :] != 0
        if self.target_transform is not None:
            img, gt = self.target_transform(img, gt)

        return img, gt, border_mask

    @staticmethod
    def get_gt_data_paths(directory: Path, data_folder_name: str, gt_folder_name: str,
                          selection: Optional[Union[int, List[str]]] = None) \
            -> List[Tuple[Path, Path, str, str, Tuple[int, int]]]:
        """
        Structure of the folder

        directory/data/ORIGINAL_FILENAME/FILE_NAME_X_Y.png
        directory/gt/ORIGINAL_FILENAME/FILE_NAME_X_Y.png


        :param directory:
        :param selection:
        :return: tuple
            (path_data_file, path_gt_file, original_image_name, (x, y))
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
            subdirectories = [x.name for x in subitems if x.is_dir()]

            if isinstance(selection, int):
                if selection < 0:
                    msg = f'Parameter "selection" is a negative integer ({selection}). ' \
                          f'Negative values are not supported!'
                    log.error(msg)
                    raise ValueError(msg)

                elif selection == 0:
                    selection = None

                elif selection > len(subdirectories):
                    msg = f'Parameter "selection" is larger ({selection}) than ' \
                          f'number of subdirectories ({len(subdirectories)}).'
                    log.error(msg)
                    raise ValueError(msg)

            elif isinstance(selection, ListConfig) or isinstance(selection, list):
                if not all(x in subdirectories for x in selection):
                    msg = f'Parameter "selection" contains a non-existing subdirectory.)'
                    log.error(msg)
                    raise ValueError(msg)

            else:
                msg = f'Parameter "selection" exists, but it is of unsupported type ({type(selection)})'
                log.error(msg)
                raise TypeError(msg)

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
