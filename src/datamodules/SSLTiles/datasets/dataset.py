from pathlib import Path

from typing import Optional, Union, List, Tuple

import numpy as np
from PIL import Image
from omegaconf import ListConfig
from torch import Tensor
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension
from torchvision.transforms import ToTensor, ToPILImage

from src.datamodules.SSLTiles.utils.misc import GT_Type
from src.datamodules.SSLTiles.utils.shuffeling import shuffle_horizontal, shuffle_vertical
from src.datamodules.RGB.datasets.full_page_dataset import DatasetRGB
from src.datamodules.utils.misc import ImageDimensions, selection_validation
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif')
log = utils.get_logger(__name__)


class DatasetSSLTiles(DatasetRGB):

    def __init__(self, path: Path, data_folder_name: str, image_dims: ImageDimensions,
                 rows: int, cols: int, horizontal_shuffle: bool, vertical_shuffle: bool, gt_type: GT_Type,
                 selection: Optional[Union[int, List[str]]] = None, image_transform=None,
                 **kwargs):
        super().__init__(path=path, data_folder_name=data_folder_name, gt_folder_name="",
                         image_dims=image_dims, selection=selection, is_test=False, image_transform=image_transform,
                         target_transform=None, twin_transform=None, **kwargs)

        if self.image_dims.height % rows != 0 or self.image_dims.width % cols != 0:
            raise ValueError('Image dimensions must be dividable by rows and cols')
        self.rows = rows
        self.cols = cols
        self.gt_type = gt_type

        if not horizontal_shuffle and not vertical_shuffle:
            raise ValueError('At least one of horizontal_shuffle or vertical_shuffle must be True')
        self.horizontal_shuffle = horizontal_shuffle
        self.vertical_shuffle = vertical_shuffle

    def __getitem__(self, index):
        data_img = self._load_data_and_gt(index)
        # create different forms of gt (class, vector with 0/1 (changed or not [just working for row==2]), matrix)
        img_tensor, gt_tensor = self._apply_transformation(data_img, None)
        return img_tensor, gt_tensor

    def _load_data_and_gt(self, index):
        return pil_loader(self.img_gt_path_list[index])

    def _apply_transformation(self, img_tensor, _):
        img_tensor, _ = super()._apply_transformation(img_tensor, None)

        # cut image in tiles and shuffle them
        new_img, gt = self._cut_image_in_tiles_and_put_together(img_tensor)

        return ToTensor()(new_img), Tensor(gt)

    @staticmethod
    def get_img_gt_path_list(directory: Path, data_folder_name: str, gt_folder_name: str = None,
                             selection: Optional[Union[int, List[str]]] = None) -> List[Path]:
        """
        Structure of the folder

        directory/data/FILE_NAME.png

        :param directory:
        :param data_folder_name:
        :param gt_folder_name:
            will not be taken into account because we dont have a gt
        :param selection:
        :return: tuple
            (path_data_file)
            it is also a list of tuples to make it inheritable
        """
        paths = []
        directory = directory.expanduser()

        path_data_root = directory / data_folder_name

        if not path_data_root.is_dir():
            log.error("folder data or gt not found in " + str(directory))

        # get all files sorted
        files_in_data_root = sorted(path_data_root.iterdir())

        # check the selection parameter
        if selection:
            selection = selection_validation(files_in_data_root, selection, full_page=True)

        counter = 0  # Counter for subdirectories, needed for selection parameter

        for path_data_file in sorted(files_in_data_root):
            counter += 1

            if selection:
                if isinstance(selection, int):
                    if counter > selection:
                        break

                elif isinstance(selection, ListConfig) or isinstance(selection, list):
                    if path_data_file.stem not in selection:
                        continue

            assert has_file_allowed_extension(path_data_file.name, IMG_EXTENSIONS), \
                'get_img_gt_path_list(): image file aligned with non-image file'

            paths.append(path_data_file)

        return paths

    def _cut_image_in_tiles_and_put_together(self, img_tensor: Tensor) -> Tuple[Image.Image, Tensor]:
        # cut image in tiles and shuffle them
        tile_dims = ImageDimensions(width=self.image_dims.width // self.cols,
                                    height=self.image_dims.height // self.rows)

        gt = np.arange(self.rows * self.cols).reshape((self.rows, self.cols))
        if self.horizontal_shuffle:
            shuffle_horizontal(gt)
        if self.vertical_shuffle:
            shuffle_vertical(gt)

        # put tiles together
        img_array = np.array(ToPILImage()(img_tensor).convert('RGB'))
        new_img_array = np.zeros(img_array.shape)
        new_img_array.fill(np.nan)

        for i in range(self.rows):
            for j in range(self.cols):
                width_start = (gt[i, j] % self.cols) * tile_dims.width
                width_end = width_start + tile_dims.width
                height_start = (gt[i, j] // self.cols) * tile_dims.height
                height_end = height_start + tile_dims.height
                new_img_array[i * tile_dims.height: (i + 1) * tile_dims.height,
                j * tile_dims.width: (j + 1) * tile_dims.width, :] = img_array[height_start:height_end,
                                                                               width_start:width_end]

        if np.isnan(np.sum(new_img_array)):
            raise ValueError('The patched image is not valid! It still contains NaN values (perhaps a patch missing)')
        return Image.fromarray(new_img_array.astype(np.uint8)), gt
