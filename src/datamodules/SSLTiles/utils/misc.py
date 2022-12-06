import itertools
from enum import Enum, auto
from pathlib import Path

import numpy as np

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir


class GT_Type(Enum):
    CLASSIFICATION = auto()
    VECTOR = auto()
    ROW_COLUMN = auto()
    FULL_IMAGE = auto()


def give_permutation():

    perms = [(0, [[0, 1], [2, 3], [4, 5]]),
             (1, [[1, 0], [2, 3], [4, 5]]),
             (2, [[0, 1], [3, 2], [4, 5]]),
             (3, [[0, 1], [2, 3], [5, 4]]),
             (4, [[1, 0], [3, 2], [4, 5]]),
             (5, [[1, 0], [2, 3], [5, 4]]),
             (6, [[0, 1], [3, 2], [5, 4]]),
             (7, [[1, 0], [3, 2], [5, 4]]),
             ]
    return perms[np.random.randint(0, len(perms))]


def validate_path_for_ssl_classification(data_dir: str):
    if data_dir is None:
        raise PathNone("Please provide the path to root dir of the dataset "
                       "(folder containing the train/val folder)")
    else:
        split_names = ['train', 'val']

        data_folder = Path(data_dir)
        if not data_folder.is_dir():
            raise PathNotDir("Please provide the path to root dir of the dataset "
                             "(folder containing the train/val folder)")
        split_folders = [d for d in data_folder.iterdir() if d.is_dir() and d.name in split_names]
        if len(split_folders) != 2:
            raise PathMissingSplitDir(f'Your path needs to contain train/val and '
                                      f'each of them a folder per class')

    return Path(data_dir)
