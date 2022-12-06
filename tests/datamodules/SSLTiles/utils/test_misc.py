import shutil

import numpy as np
import pytest

from src.datamodules.SSLTiles.utils.misc import give_permutation, GT_type, validate_path_for_ssl_classification
from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir_classification


def test_give_permutation():
    np.random.seed(42)
    perm = give_permutation()
    assert perm[0] == 6
    assert perm[1] == [[0, 1], [3, 2], [5, 4]]


def test_validate_path_for_ssl_classification(data_dir_classification):
    data_dir = validate_path_for_ssl_classification(data_dir=str(data_dir_classification))
    assert data_dir == data_dir_classification


def test_validate_path_for_ssl_classification_path_none(data_dir_classification):
    with pytest.raises(PathNone):
        validate_path_for_ssl_classification(data_dir=None)


def test_validate_path_for_ssl_classification_split_missing(data_dir_classification):
    shutil.rmtree(data_dir_classification / 'val')
    with pytest.raises(PathMissingSplitDir):
        validate_path_for_ssl_classification(data_dir=data_dir_classification)


def test_validate_path_for_ssl_classification_not_dir(data_dir_classification):
    with pytest.raises(PathNotDir):
        validate_path_for_ssl_classification(data_dir=data_dir_classification / '0' / 'fmb-cb-55-052r.png')
