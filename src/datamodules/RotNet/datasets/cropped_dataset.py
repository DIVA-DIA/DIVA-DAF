"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
import torchvision.transforms.functional
from omegaconf import ListConfig
from torch import is_tensor
from torchvision.transforms import ToTensor

from src.datamodules.DivaHisDB.datasets.cropped_dataset import CroppedHisDBDataset
from src.datamodules.RotNet.utils.misc import has_extension, pil_loader
from src.utils import utils

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
ROTATION_ANGLES = [0, 90, 180, 270]

log = utils.get_logger(__name__)


class CroppedRotNet(CroppedHisDBDataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png
    """

    def __init__(self, path: Path, data_folder_name: str = 'data', gt_folder_name: str = 'gt',
                 selection: Optional[Union[int, List[str]]] = None,
                 is_test=False, image_transform=None, **kwargs):
        """
        #TODO doc
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

        super(CroppedRotNet, self).__init__(path=path, data_folder_name=data_folder_name, gt_folder_name=gt_folder_name,
                                            selection=selection,
                                            is_test=is_test, image_transform=image_transform,
                                            target_transform=None, twin_transform=None,
                                            classes=None, **kwargs)

    def __getitem__(self, index):
        data_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, index=index)
        return img, gt

    def _load_data_and_gt(self, index):
        data_img = pil_loader(self.img_paths_per_page[index])
        return data_img

    def _apply_transformation(self, img, index):
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
            img, gt = self.twin_transform(img, None)

        if self.image_transform is not None:
            # perform transformations
            img = self.image_transform(img)

        if not is_tensor(img):
            img = ToTensor()(img)

        target_class = index % len(ROTATION_ANGLES)
        rotation_angle = ROTATION_ANGLES[target_class]
        hot_hot_encoded = np.zeros(len(ROTATION_ANGLES))
        hot_hot_encoded[target_class] = 1

        img = torchvision.transforms.functional.rotate(img=img, angle=rotation_angle)

        return img, torch.LongTensor(hot_hot_encoded)

    @staticmethod
    def get_gt_data_paths(directory: Path, data_folder_name: str = 'data', gt_folder_name: str = 'gt',
                          selection: Optional[Union[int, List[str]]] = None) \
            -> List[Path]:
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
                if has_extension(path_data_subdir.name, IMG_EXTENSIONS):
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

            for path_data_file in sorted(path_data_subdir.iterdir()):
                if has_extension(path_data_file.name, IMG_EXTENSIONS):
                    paths.append(path_data_file)
                    paths.append(path_data_file)
                    paths.append(path_data_file)
                    paths.append(path_data_file)

        return paths
