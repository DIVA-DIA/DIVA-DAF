import pytest
import shutil

from src.datamodules.RotNet.utils.misc import validate_path_for_self_supervised
from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir, PathMissingDirinSplitDir
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped


def test_validate_path_for_self_supervised(data_dir_cropped):
    output = validate_path_for_self_supervised(data_dir=data_dir_cropped, data_folder_name='data')
    assert output == data_dir_cropped


def test_validate_path_for_self_supervised_dir_none(data_dir_cropped):
    with pytest.raises(PathNone):
        validate_path_for_self_supervised(data_dir=None, data_folder_name='data')


def test_validate_path_for_self_supervised_data_not_dir(data_dir_cropped):
    with pytest.raises(PathNotDir):
        validate_path_for_self_supervised(data_dir=data_dir_cropped / 'test.jpg', data_folder_name='data')


def test_validate_path_for_self_supervised_data_missing_split(data_dir_cropped):
    shutil.rmtree(data_dir_cropped / 'test')
    with pytest.raises(PathMissingSplitDir):
        validate_path_for_self_supervised(data_dir=data_dir_cropped, data_folder_name='data')
