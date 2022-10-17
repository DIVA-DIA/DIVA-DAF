from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from PIL import Image
from omegaconf import ListConfig

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir, PathMissingDirinSplitDir
from src.utils import utils

log = utils.get_logger(__name__)


@dataclass
class ImageDimensions:
    width: int
    height: int


def _get_argmax(output: Union[torch.Tensor, np.ndarray], dim=1):
    """
    takes the biggest value from a pixel across all classes
    :param output: (Batch_size x num_classes x W x H)
        matrix with the given attributes
    :return: (Batch_size x W x H)
        matrix with the hisdb class number for each pixel
    """
    if isinstance(output, torch.Tensor):
        return torch.argmax(output, dim=dim)
    if isinstance(output, np.ndarray):
        return np.argmax(output, axis=dim)
    return output


def validate_path_for_segmentation(data_dir, data_folder_name: str, gt_folder_name: str,
                                   split_name: Union[str, List[str]]):
    """
    Checks if the data_dir folder has the following structure:

    {data_dir}
        - {train_folder_name}
            - {data_folder_name}
            - {gt_folder_name}
        - {val_folder_name}
            - {data_folder_name}
            - {gt_folder_name}
        - {test_folder_name}
            - {data_folder_name}
            - {gt_folder_name}


    :param split_name:
    :param data_dir:
    :param data_folder_name:
    :param gt_folder_name:
    :return:
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
    if filename not in current_list:
        return filename
    for i in range(len(current_list)):
        new_filename = f'{filename}_{i}'
        if new_filename not in current_list:
            return new_filename
    else:
        log.error('Unexpected error: Did not find new filename that is not a duplicate!')
        raise AssertionError


def selection_validation(files_in_data_root: List[Path], selection, full_page: bool):
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
                msg = f'Parameter "selection" contains a non-existing file names.)'
                log.error(msg)
                raise ValueError(msg)
        else:
            if not all(x in subdirectories for x in selection):
                msg = f'Parameter "selection" contains a non-existing subdirectory.)'
                log.error(msg)
                raise ValueError(msg)

    else:
        msg = f'Parameter "selection" exists, but it is of unsupported type ({type(selection)})'
        log.error(msg)
        raise TypeError(msg)

    return selection


def get_image_dims(data_gt_path_list, **kwargs):
    if isinstance(data_gt_path_list[0], tuple):
        img = Image.open(data_gt_path_list[0][0]).convert('RGB')
    else:
        img = Image.open(data_gt_path_list[0]).convert('RGB')

    image_dims = ImageDimensions(width=img.width, height=img.height)

    return image_dims
