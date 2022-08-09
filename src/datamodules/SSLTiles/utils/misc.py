from enum import Enum, auto

import numpy as np


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
