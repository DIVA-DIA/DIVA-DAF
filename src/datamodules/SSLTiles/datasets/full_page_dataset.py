from pathlib import Path

from typing import Optional, Union, List, Tuple

from omegaconf import ListConfig
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension

from src.datamodules.RGB.datasets.full_page_dataset import DatasetRGB
from src.datamodules.utils.misc import ImageDimensions, selection_validation
from src.utils import utils

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif')
log = utils.get_logger(__name__)


class DatasetSSLTiles(DatasetRGB):

    def __init__(self, path: Path, data_folder_name: str, gt_folder_name: str, image_dims: ImageDimensions,
                 rows: int, cols: int, horizontal_shuffle: bool, vertical_shuffle: bool,
                 selection: Optional[Union[int, List[str]]] = None, image_transform=None,
                 **kwargs):
        super().__init__(path=path, data_folder_name=data_folder_name, gt_folder_name=gt_folder_name,
                         image_dims=image_dims, selection=selection, is_test=False, image_transform=image_transform,
                         target_transform=None, twin_transform=None, **kwargs)
        self.rows = rows
        self.cols = cols
        self.horizontal_shuffle = horizontal_shuffle
        self.vertical_shuffle = vertical_shuffle

    def __getitem__(self, index):
        data_img = self._load_data_and_gt(index)
        # create gt_img
        return super().__getitem__(index)

    def _load_data_and_gt(self, index):
        return pil_loader(self.img_gt_path_list[index])

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


