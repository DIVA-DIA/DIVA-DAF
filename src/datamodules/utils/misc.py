from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir, PathMissingDirinSplitDir


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


@dataclass
class ImageDimensions:
    width: int
    height: int