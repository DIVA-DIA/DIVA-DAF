import pytest
from pytest import fixture

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir, PathMissingDirinSplitDir
from src.datamodules.DivaHisDB.utils.misc import validate_path_for_segmentation


@fixture
def path_missing_split(tmp_path):
    list_splits = ['train', 'test']

    for split_name in list_splits:
        split_path = tmp_path / split_name
        split_path.mkdir()

    return tmp_path


@fixture
def path_missing_subfolder(tmp_path):
    list_splits_good = ['train', 'val']
    list_types_good = ['data', 'gt']
    list_splits_bad = ['test']
    list_types_bad = ['gt']

    for split_name in list_splits_good:
        split_path = tmp_path / split_name
        split_path.mkdir()
        for type_name in list_types_good:
            type_path = split_path / type_name
            type_path.mkdir()

    for split_name in list_splits_bad:
        split_path = tmp_path / split_name
        split_path.mkdir()
        for type_name in list_types_bad:
            type_path = split_path / type_name
            type_path.mkdir()

    return tmp_path


def test_validate_path_none():
    with pytest.raises(PathNone):
        validate_path_for_segmentation(data_dir=None)


def test_validate_path_not_dir(tmp_path):
    tmp_file = tmp_path / "newfile"
    tmp_file.touch()
    with pytest.raises(PathNotDir):
        validate_path_for_segmentation(data_dir=tmp_file)


def test_validate_path_missing_split(path_missing_split):
    with pytest.raises(PathMissingSplitDir):
        validate_path_for_segmentation(data_dir=path_missing_split)


def test_validate_path_missing_subfolder(path_missing_subfolder):
    with pytest.raises(PathMissingDirinSplitDir):
        validate_path_for_segmentation(data_dir=path_missing_subfolder)
