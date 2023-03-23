from pathlib import Path

import numpy as np
import pytest
import torch

from src.datamodules.utils.exceptions import PathNone, PathNotDir, PathMissingSplitDir, PathMissingDirinSplitDir
from src.datamodules.utils.misc import validate_path_for_segmentation, _get_argmax, get_output_file_list, \
    find_new_filename, selection_validation, get_image_dims
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped, data_dir


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
        validate_path_for_segmentation(data_dir=None, data_folder_name='data', gt_folder_name='gt',
                                       split_name='test')


def test_validate_path_not_dir(tmp_path):
    tmp_file = tmp_path / "newfile"
    tmp_file.touch()
    with pytest.raises(PathNotDir):
        validate_path_for_segmentation(data_dir=tmp_file, data_folder_name='data', gt_folder_name='gt',
                                       split_name='train')


def test_validate_path_missing_split(path_missing_split):
    with pytest.raises(PathMissingSplitDir):
        validate_path_for_segmentation(data_dir=path_missing_split, data_folder_name='data', gt_folder_name='gt',
                                       split_name='something')


def test_validate_path_missing_subfolder(path_missing_subfolder):
    with pytest.raises(PathMissingDirinSplitDir):
        validate_path_for_segmentation(data_dir=path_missing_subfolder, data_folder_name='data', gt_folder_name='gt',
                                       split_name='test')


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


@pytest.fixture
def get_test_data_cropped_page(data_dir_cropped):
    return sorted(list(list(list(filter(Path.is_dir, list(data_dir_cropped.iterdir())))[0].iterdir()))[0].iterdir())


@pytest.fixture
def get_test_data_full_page(data_dir):
    return sorted(list(list(list(filter(Path.is_dir, list(data_dir.iterdir())))[0].iterdir())[0].iterdir()))


def test_selection_validation_int_working_full_page(get_test_data_full_page):
    selection = 1
    selection_r = selection_validation(files_in_data_root=get_test_data_full_page, selection=selection,
                                       full_page=True)
    assert selection_r == 1
    assert selection == selection_r


def test_selection_validation_int_zero_full_page(get_test_data_full_page):
    selection = 0
    selection_r = selection_validation(files_in_data_root=get_test_data_full_page, selection=selection,
                                       full_page=True)
    assert selection_r is None


def test_selection_validation_int_negative_full_page(get_test_data_full_page):
    selection = -4
    with pytest.raises(ValueError):
        selection_validation(files_in_data_root=get_test_data_full_page, selection=selection,
                             full_page=True)


def test_selection_validation_int_too_big_full_page(get_test_data_full_page):
    selection = 4
    with pytest.raises(ValueError):
        selection_validation(files_in_data_root=get_test_data_full_page, selection=selection,
                             full_page=True)


def test_selection_validation_int_working_cropped_page(get_test_data_cropped_page):
    selection = 1
    selection_r = selection_validation(files_in_data_root=get_test_data_cropped_page, selection=selection,
                                       full_page=False)
    assert selection_r == 1
    assert selection == selection_r


def test_selection_validation_int_zero_cropped_page(get_test_data_cropped_page):
    selection = 0
    selection_r = selection_validation(files_in_data_root=get_test_data_cropped_page, selection=selection,
                                       full_page=False)
    assert selection_r is None


def test_selection_validation_int_negative_cropped_page(get_test_data_cropped_page):
    selection = -4
    with pytest.raises(ValueError):
        selection_validation(files_in_data_root=get_test_data_cropped_page, selection=selection,
                             full_page=False)


def test_selection_validation_int_too_big_cropped_page(get_test_data_cropped_page):
    selection = 4
    with pytest.raises(ValueError):
        selection_validation(files_in_data_root=get_test_data_cropped_page, selection=selection,
                             full_page=False)


def test_selection_validation_unsupported_type(get_test_data_cropped_page):
    selection = 'test'
    with pytest.raises(TypeError):
        selection_validation(files_in_data_root=get_test_data_cropped_page, selection=selection,
                             full_page=False)


def test_selection_validation_list_full_page_wrong_name(get_test_data_full_page):
    selection = ['not_in_list']
    with pytest.raises(ValueError):
        selection_validation(files_in_data_root=get_test_data_full_page, selection=selection,
                             full_page=True)


def test_selection_validation_list_full_page(get_test_data_full_page):
    selection = ['e-codices_fmb-cb-0055_0098v_max']
    selection_r = selection_validation(files_in_data_root=get_test_data_full_page, selection=selection,
                                       full_page=True)
    assert len(selection_r) == 1


def test_selection_validation_list_cropped_page_wrong_name(get_test_data_cropped_page):
    selection = ['not_in_list']
    with pytest.raises(ValueError):
        selection_validation(files_in_data_root=get_test_data_cropped_page, selection=selection,
                             full_page=False)


def test_selection_validation_list_cropped_page(get_test_data_cropped_page):
    selection = ['e-codices_fmb-cb-0055_0098v_max']
    selection_r = selection_validation(files_in_data_root=get_test_data_cropped_page, selection=selection,
                                       full_page=False)
    assert len(selection_r) == 1


def test_get_image_dims_list_paths(get_test_data_full_page):
    img_dims = get_image_dims(get_test_data_full_page)
    assert img_dims.width == 487
    assert img_dims.height == 649


def test_get_image_dims_list_tuples(get_test_data_full_page):
    img_dims = get_image_dims([(get_test_data_full_page[0], get_test_data_full_page[0])])
    assert img_dims.width == 487
    assert img_dims.height == 649
