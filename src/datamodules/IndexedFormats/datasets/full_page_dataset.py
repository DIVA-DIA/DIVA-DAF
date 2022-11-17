"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import torch.utils.data as data
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
    """

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str,
                 image_dims: ImageDimensions, is_test=False,
                 selection: Optional[Union[int, List[str]]] = None,
                 image_transform=None):
        """
        Parameters
        ----------
        path: Path
            Path to dataset folder (train / val / test)
        data_folder_name: string
            name of the folder inside of the train/val/test that contains the images
        gt_folder_name: string
            name of the folder in train/val/test containing the ground truth images
        image_dims: ImageDimensions
            Dimension of the image(s)
        selection: int or list(str)
            number of files or list of files that should be taken into account for this split.
        is_test: bool
            Indicate if the split is the test split
        image_transform : callable
        target_transform : callable
        twin_transform : callable
            A function to load an image given its path.
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

    def __getitem__(self, index):
        data_img, gt_img = self._load_data_and_gt(index=index)
        img, gt = self._apply_transformation(data_img, gt_img)
        assert img.shape[-2:] == gt.shape[-2:]

        return img, gt

    def _load_data_and_gt(self, index):
        data_img = pil_loader(self.img_gt_path_list[index][0])
        gt_img = pil_loader_gif(self.img_gt_path_list[index][1])

        assert data_img.height == self.image_dims.height and data_img.width == self.image_dims.width
        assert gt_img.height == self.image_dims.height and gt_img.width == self.image_dims.width

        return data_img, gt_img

    def _apply_transformation(self, img, gt):
        if self.image_transform is not None:
            # perform transformations
            img = self.image_transform(img)

        if not is_tensor(img):
            img = ToTensor()(img)

        gt = ToTensor()(np.asarray(gt))

        return img, gt

    @staticmethod
    def get_img_gt_path_list(directory: Path, data_folder_name: str, gt_folder_name: str,
                             selection: Optional[Union[int, List[str]]] = None) \
            -> List[Tuple[Path, Path]]:
        """
        Structure of the folder

        directory/data/FILE_NAME.png
        directory/gt/FILE_NAME.gif

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
                   has_file_allowed_extension(path_gt_file.name, GT_EXTENSION), \
                'get_img_gt_path_list(): image file aligned with non-image file'

            # assert path_data_file.stem == path_gt_file.stem, \
            #     'get_img_gt_path_list(): mismatch between data filename and gt filename'
            paths.append((path_data_file, path_gt_file))

        return paths
