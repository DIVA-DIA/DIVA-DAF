import shutil

import numpy as np
import pytest

from src.datamodules.Classification.utils.misc import validate_path_for_classification
from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir_classification


def test_validate_path_for_ssl_classification(data_dir_classification):
    data_dir = validate_path_for_classification(data_dir=str(data_dir_classification))
    assert data_dir == data_dir_classification


def test_validate_path_for_ssl_classification_path_none(data_dir_classification):
    with pytest.raises(PathNone):
        validate_path_for_classification(data_dir=None)


def test_validate_path_for_ssl_classification_split_missing(data_dir_classification):
    shutil.rmtree(data_dir_classification / 'val')
    with pytest.raises(PathMissingSplitDir):
        validate_path_for_classification(data_dir=data_dir_classification)


def test_validate_path_for_ssl_classification_not_dir(data_dir_classification):
    with pytest.raises(PathNotDir):
        validate_path_for_classification(data_dir=data_dir_classification / '0' / 'fmb-cb-55-052r.png')
