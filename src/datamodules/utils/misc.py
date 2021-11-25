from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

import numpy as np
import torch

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


def validate_path_for_segmentation(data_dir, data_folder_name: str, gt_folder_name: str):
    """
    Checks if the data_dir folder has the following structure:

    {data_dir}
        - train
            - {data_folder_name}
            - {gt_folder_name}
        - val
            - {data_folder_name}
            - {gt_folder_name}
        - test
            - {data_folder_name}
            - {gt_folder_name}


    :param data_dir:
    :param data_folder_name:
    :param gt_folder_name:
    :return:
    """
    if data_dir is None:
        raise PathNone("Please provide the path to root dir of the dataset "
                       "(folder containing the train/val/test folder)")
    else:
        split_names = ['train', 'val', 'test']
        type_names = [data_folder_name, gt_folder_name]

        data_folder = Path(data_dir)
        if not data_folder.is_dir():
            raise PathNotDir("Please provide the path to root dir of the dataset "
                             "(folder containing the train/val/test folder)")
        split_folders = [d for d in data_folder.iterdir() if d.is_dir() and d.name in split_names]
        if len(split_folders) != 3:
            raise PathMissingSplitDir(f'Your path needs to contain train/val/test and '
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
        log.warn(f'Duplicate filenames in output list. '
                 f'Output filenames have been changed to be unique. Duplicates:\n'
                 f'{duplicate_filenames}')

    return output_list


def find_new_filename(filename: str, current_list: List[str]) -> str:
    for i in range(len(current_list)):
        new_filename = f'{filename}_{i}'
        if new_filename not in current_list:
            return new_filename
    else:
        log.error('Unexpected error: Did not find new filename that is not a duplicate!')
        raise AssertionError