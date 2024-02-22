import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict, Tuple, Any

import numpy as np
import torch
from PIL import Image
from omegaconf import ListConfig

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir, PathMissingDirinSplitDir
from src.utils import utils

log = utils.get_logger(__name__)


@dataclass
class ImageDimensions:
    """
    Dataclass to store the dimensions of an image

    :param width: Width of the image
    :type width: int
    :param height: Height of the image
    :type height: int
    """
    width: int
    height: int


def _get_argmax(output: Union[torch.Tensor, np.ndarray], dim=1) -> Union[torch.Tensor, np.ndarray]:
    """
    takes the biggest value from a pixel across all classes

    :param output: (Batch_size x num_classes x W x H)
        matrix with the given attributes
    :type output: torch.Tensor or np.ndarray
    :param dim: dimension to take the argmax over
    :type dim: int
    :returns: (Batch_size x W x H)
        matrix with the hisdb class number for each pixel
    :type output: torch.Tensor or np.ndarray
    """
    if isinstance(output, torch.Tensor):
        return torch.argmax(output, dim=dim)
    if isinstance(output, np.ndarray):
        return np.argmax(output, axis=dim)
    return output


def validate_path_for_segmentation(data_dir: str, data_folder_name: str, gt_folder_name: str,
                                   split_name: Union[str, List[str]]) -> Path:
    """
    Checks if the data_dir folder has the following structure::

    data_dir
        ├── train_folder_name
        │   ├── data_folder_name
        │   └── gt_folder_name
        ├── val_folder_name
        │   ├── data_folder_name
        │   └── gt_folder_name
        └── test_folder_name
            ├── data_folder_name
            └── gt_folder_name

    :param data_dir: Path to the root dir of the dataset
    :type data_dir: str
    :param data_folder_name: Name of the data folder
    :type data_folder_name: str
    :param gt_folder_name: Name of the gt folder
    :type gt_folder_name: str
    :param split_name: Name of the split folder (train/val/test)
    :type split_name: str

    :returns: Path to the data_dir
    :rtype: Path
    """
    if data_dir is None:
        raise PathNone("Please provide the path to root dir of the dataset "
                       "(folder containing the split(train/val/test) folders)")
    if isinstance(split_name, str):
        split_names = [split_name]
    else:
        split_names = split_name
    type_names = [data_folder_name, gt_folder_name]

    data_folder = Path(data_dir)
    if not data_folder.is_dir():
        raise PathNotDir("Please provide the path to root dir of the dataset "
                         "(folder containing the split(train/val/test) folder)")
    split_folders = [d for d in data_folder.iterdir() if d.is_dir() and d.name in split_names]
    if len(split_folders) != len(split_names):
        raise PathMissingSplitDir(
            f'Your path needs to contain the folder(s) "{split_name}"'
            f'each of them a folder {data_folder_name} and {gt_folder_name}')

    # check if we have train/test/val
    for split in split_folders:
        type_folders = [d for d in split.iterdir() if d.is_dir() and d.name in type_names]
        # check if we have data/gt
        if len(type_folders) != 2:
            raise PathMissingDirinSplitDir(f'Folder {split.name} does not contain a {data_folder_name} '
                                           f'and {gt_folder_name} folder')
    return Path(data_dir)


def get_output_file_list(image_path_list: List[Path]) -> List[str]:
    """
    Creates a list of output filenames from a list of image paths.
    If there are duplicate filenames, the duplicates are renamed to be unique.

    :param image_path_list: List of image paths
    :type image_path_list: List[Path]
    :returns: List of output filenames
    :rtype: List[str]
    """

    duplicate_filenames = []
    output_list = []
    for p in image_path_list:
        filename = p.stem
        if filename not in output_list:
            output_list.append(filename)
        else:
            duplicate_filenames.append(filename)
            new_filename = find_new_filename(filename=filename, current_list=output_list)
            assert new_filename is not None and len(new_filename) > 0
            assert new_filename not in output_list
            output_list.append(new_filename)

    assert len(image_path_list) == len(output_list)

    if len(duplicate_filenames) > 0:
        log.warning(f'Duplicate filenames in output list. '
                    f'Output filenames have been changed to be unique. Duplicates:\n'
                    f'{duplicate_filenames}')

    return output_list


def find_new_filename(filename: str, current_list: List[str]) -> str:
    """
    Finds a new filename that is not in the current list.
    If the filename is not in the list, it is returned.
    If the filename is in the list, a number is appended to the filename until it is unique.

    :param filename: Filename to check
    :type filename: str
    :param current_list: List of filenames to check against
    :type current_list: List[str]
    :returns: New filename that is not in the current list
    :rtype: str
    """
    if filename not in current_list:
        return filename
    for i in range(len(current_list)):
        new_filename = f'{filename}_{i}'
        if new_filename not in current_list:
            return new_filename

    log.error('Unexpected error: Did not find new filename that is not a duplicate!')
    raise AssertionError


def selection_validation(files_in_data_root: List[Path], selection: Union[int, List[str], ListConfig],
                         full_page: bool) -> Union[int, List[str], ListConfig]:
    """
    Validates the selection parameter for the segmentation dataset.
    If selection is an integer, it is checked if it is in the range of the number of files.
    If selection is a list, it is checked if all elements are in the list of files.
    If selection is None, it is returned.

    :param files_in_data_root: List of files in the data root directory
    :type files_in_data_root: List[Path]
    :param selection: Selection parameter
    :type selection: Union[int, List[str], ListConfig]
    :param full_page: If True, the selection parameter is used to select a page.
                        If False, the selection parameter is used to select a subdirectory.
    :type full_page: bool
    :returns: Validated selection parameter
    :rtype: Union[int, List[str], ListConfig]
    """
    if not full_page:
        subdirectories = [x.name for x in files_in_data_root if x.is_dir()]

    if isinstance(selection, int):

        if selection < 0:
            msg = f'Parameter "selection" is a negative integer ({selection}). ' \
                  f'Negative values are not supported!'
            log.error(msg)
            raise ValueError(msg)

        elif selection == 0:
            selection = None

        elif (selection > len(files_in_data_root) and full_page) or (not full_page and selection > len(subdirectories)):
            msg = f'Parameter "selection" is larger ({selection}) than ' \
                  f'number of files ({len(files_in_data_root)}).'
            log.error(msg)
            raise ValueError(msg)

    elif isinstance(selection, ListConfig) or isinstance(selection, list):
        if full_page:
            if not all(x in [f.stem for f in files_in_data_root] for x in selection):
                msg = 'Parameter "selection" contains a non-existing file names.)'
                log.error(msg)
                raise ValueError(msg)
        else:
            if not all(x in subdirectories for x in selection):
                msg = 'Parameter "selection" contains a non-existing subdirectory.)'
                log.error(msg)
                raise ValueError(msg)

    else:
        msg = f'Parameter "selection" exists, but it is of unsupported type ({type(selection)})'
        log.error(msg)
        raise TypeError(msg)

    return selection


def get_image_dims(data_gt_path_list) -> ImageDimensions:
    """
    Returns the image dimensions of the first image in the list.

    :param data_gt_path_list: List of image paths
    :type data_gt_path_list: List[Path]
    :returns: Image dimensions of the first image in the list
    :rtype: ImageDimensions
    """
    if isinstance(data_gt_path_list[0], tuple):
        img = Image.open(data_gt_path_list[0][0]).convert('RGB')
    else:
        img = Image.open(data_gt_path_list[0]).convert('RGB')

    image_dims = ImageDimensions(width=img.width, height=img.height)

    return image_dims


def pil_loader_gif(path: Path) -> Image:
    """
    Loads a gif image using PIL.

    :param path: Path to the image
    :type path: Path
    :returns: Image loaded in palette mode
    :rtype: Image
    """
    with open(path, "rb") as f:
        gt_img = Image.open(f)
        return gt_img.convert('P')


def save_json(analytics: Dict, analytics_path: Path):
    """
    Saves the analytics dict to a json file.

    :param analytics: The analytics dict that should be saved
    :type analytics: Dict
    :param analytics_path: Path to the json file
    :type analytics_path: Path
    """

    try:
        with analytics_path.open(mode='w') as f:
            json.dump(obj=analytics, fp=f)
    except IOError:
        print(f'WARNING: No permissions to write analytics file ({analytics_path})')


def check_missing_analytics(analytics_path_gt: Path, expected_keys_gt: List[str]) -> Tuple[Dict[str, Any], bool]:
    """
    Check if the analytics file for the ground truth is missing and if it is complete. If its is present, it will be
    loaded and the contained keys checked for completeness.

    :param analytics_path_gt: Path where the analytics file should be
    :type analytics_path_gt: Path
    :param expected_keys_gt: List of expected keys in the analytics file
    :type expected_keys_gt: List[str]
    :return: Tuple of the loaded analytics and a boolean indicating if the analytics file is missing
    :rtype: Tuple[Dict[str, Any], bool]
    """
    missing_analytics = True
    analytics = None

    if analytics_path_gt.exists():
        with analytics_path_gt.open(mode='r') as f:
            analytics = json.load(fp=f)
        # check if analytics file is complete
        if all(k in analytics for k in expected_keys_gt):
            missing_analytics = False
    return analytics, missing_analytics
