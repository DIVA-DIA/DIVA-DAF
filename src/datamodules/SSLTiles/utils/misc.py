import itertools
from enum import Enum, auto

import numpy as np


class GT_Type(Enum):
    CLASSIFICATION = auto()
    VECTOR = auto()
    ROW_COLUMN = auto()
    FULL_IMAGE = auto()


def give_permutation(rows: int, cols: int, gt_type: GT_Type, horizontal_shuffle: bool = True,
                     vertical_shuffle: bool = True):

    gt_structure = np.arange(rows * cols).reshape((rows, cols))
    if horizontal_shuffle:
        perms = itertools.permutations(np.arange(rows))
    if vertical_shuffle:
        raise NotImplementedError("vertical_shuffle not implemented yet")

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
