import numpy as np

from src.datamodules.SSLTiles.utils.misc import give_permutation, GT_Type


def test_give_permutation():
    np.random.seed(42)
    perm = give_permutation(rows=3, cols=2, gt_type=GT_Type.ROW_COLUMN, horizontal_shuffle=True, vertical_shuffle=False)
    assert perm[0] == 6
    assert perm[1] == [[0, 1], [3, 2], [5, 4]]
