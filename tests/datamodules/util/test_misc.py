from pathlib import Path

import numpy as np
import pytest
import torch

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir, PathMissingDirinSplitDir
from src.datamodules.utils.misc import validate_path_for_segmentation, _get_argmax, get_output_file_list, \
    find_new_filename


@pytest.fixture
def path_missing_split(tmp_path):
    list_splits = ['train', 'test']

    for split_name in list_splits:
        split_path = tmp_path / split_name
        split_path.mkdir()

    return tmp_path


@pytest.fixture
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
        validate_path_for_segmentation(data_dir=None, data_folder_name='data', gt_folder_name='gt')


def test_validate_path_not_dir(tmp_path):
    tmp_file = tmp_path / "newfile"
    tmp_file.touch()
    with pytest.raises(PathNotDir):
        validate_path_for_segmentation(data_dir=tmp_file, data_folder_name='data', gt_folder_name='gt')


def test_validate_path_missing_split(path_missing_split):
    with pytest.raises(PathMissingSplitDir):
        validate_path_for_segmentation(data_dir=path_missing_split, data_folder_name='data', gt_folder_name='gt')


def test_validate_path_missing_subfolder(path_missing_subfolder):
    with pytest.raises(PathMissingDirinSplitDir):
        validate_path_for_segmentation(data_dir=path_missing_subfolder, data_folder_name='data', gt_folder_name='gt')


def test__get_argmax_tensor():
    input_tensor = torch.tensor([[[[0.0130, 0.1727],
                                   [0.0597, 0.2523]],
                                  [[0.3631, 0.9832],
                                   [0.4938, 0.2668]],
                                  [[0.3995, 0.8884],
                                   [0.9181, 0.7162]]]])
    output_tensor = _get_argmax(input_tensor)
    assert torch.equal(output_tensor, torch.tensor([[[2, 1], [2, 2]]]))
    assert output_tensor.shape == torch.Size([1, 2, 2])


def test__get_argmax_np():
    input_np = np.array([[[[0.0130, 0.1727],
                           [0.0597, 0.2523]],
                          [[0.3631, 0.9832],
                           [0.4938, 0.2668]],
                          [[0.3995, 0.8884],
                           [0.9181, 0.7162]]]])
    output_np = _get_argmax(input_np)
    assert np.array_equal(output_np, np.array([[[2, 1], [2, 2]]]))
    assert output_np.shape == (1, 2, 2)


def test_get_output_file_list_no_duplicates():
    image_path_list = [Path('second/test1.jpg'), Path('test2.jpg'), Path('test3.jpg')]
    output_list = get_output_file_list(image_path_list=image_path_list)
    assert len(output_list) == 3
    assert output_list == ['test1', 'test2', 'test3']


def test_get_output_file_list_with_duplicates(caplog):
    image_path_list = [Path('second/test1.jpg'), Path('first/test1.jpg'), Path('test3.jpg')]
    output_list = get_output_file_list(image_path_list=image_path_list)
    warning = f"Duplicate filenames in output list. " \
              f"Output filenames have been changed to be unique. Duplicates:\n" \
              f"['test1']"
    assert len(output_list) == 3
    assert output_list == ['test1', 'test1_0', 'test3']
    assert warning in caplog.text


def test_find_new_filename_no_duplicate():
    current_list = ['test1', 'test2', 'test3']
    filename = 'test4'
    new_file_name = find_new_filename(current_list=current_list, filename=filename)
    assert new_file_name == 'test4'


def test_find_new_filename_duplicate():
    current_list = ['test1', 'test2', 'test3']
    filename = 'test3'
    new_file_name = find_new_filename(current_list=current_list, filename=filename)
    assert new_file_name == 'test3_0'

