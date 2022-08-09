import numpy as np

from src.datamodules.SSLTiles.utils.misc import give_permutation


def test_give_permutation():
    np.random.seed(42)
    perm = give_permutation()
    assert perm[0] == 6
    assert perm[1] == [[0, 1], [3, 2], [5, 4]]
