from pathlib import PosixPath

import pytest
import torch

from src.datamodules.RGB.datasets.cropped_dataset import CroppedDatasetRGB
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

DATA_FOLDER_NAME = 'data'
GT_FOLDER_NAME = 'gt'
DATASET_PREFIX = 'e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max'


@pytest.fixture
def dataset_test(data_dir_cropped):
    return CroppedDatasetRGB(path=data_dir_cropped / 'test', data_folder_name=DATA_FOLDER_NAME,
                             gt_folder_name=GT_FOLDER_NAME, is_test=True)


@pytest.fixture
def dataset_val(data_dir_cropped):
    return CroppedDatasetRGB(path=data_dir_cropped / 'val', data_folder_name=DATA_FOLDER_NAME,
                             gt_folder_name=GT_FOLDER_NAME)


@pytest.fixture
def dataset_train(data_dir_cropped):
    return CroppedDatasetRGB(path=data_dir_cropped / 'train', data_folder_name=DATA_FOLDER_NAME,
                             gt_folder_name=GT_FOLDER_NAME)


def test___len__(data_dir_cropped):
    dataset = CroppedDatasetRGB(path=data_dir_cropped / 'train', data_folder_name=DATA_FOLDER_NAME,
                                gt_folder_name=GT_FOLDER_NAME)
    assert len(dataset) == 12


def test__load_data_and_gt(dataset_train):
    data_img, gt_img = dataset_train._load_data_and_gt(0)
    assert data_img.size == (300, 300)
    assert gt_img.size == (300, 300)


def test__get_train_val_items_train(dataset_train):
    img, gt = dataset_train._get_train_val_items(index=0)
    assert img.shape == torch.Size([3, 300, 300])
    assert gt.shape == torch.Size([3, 300, 300])


def test__get_train_val_items_val(dataset_val):
    img, gt = dataset_val._get_train_val_items(index=0)
    assert img.shape == torch.Size([3, 300, 300])
    assert gt.shape == torch.Size([3, 300, 300])


def test__get_train_val_items_test(dataset_test):
    img, gt, index = dataset_test._get_test_items(index=0)
    assert img.shape == torch.Size([3, 256, 256])
    assert gt.shape == torch.Size([3, 256, 256])
    assert index == 0


def test_dataset_train_selection_int_error_negative(data_dir_cropped):
    with pytest.raises(ValueError):
        CroppedDatasetRGB.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                            data_folder_name=DATA_FOLDER_NAME, gt_folder_name=GT_FOLDER_NAME,
                                            selection=-2)


def test_dataset_train_selection_int_error(data_dir_cropped):
    with pytest.raises(ValueError):
        CroppedDatasetRGB.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                            data_folder_name=DATA_FOLDER_NAME, gt_folder_name=GT_FOLDER_NAME,
                                            selection=2)


def test_dataset_train_selection_int(data_dir_cropped, get_train_file_names):
    files_from_method = CroppedDatasetRGB.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                                            data_folder_name=DATA_FOLDER_NAME,
                                                            gt_folder_name=GT_FOLDER_NAME, selection=1)
    assert len(files_from_method) == 12
    assert files_from_method == get_train_file_names


def test_dataset_train_selection_list(data_dir_cropped, get_train_file_names):
    files_from_method = CroppedDatasetRGB.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                                  data_folder_name=DATA_FOLDER_NAME,
                                                  gt_folder_name=GT_FOLDER_NAME,
                                                  selection=['e-codices_fmb-cb-0055_0098v_max'])
    assert len(files_from_method) == 12
    assert files_from_method == get_train_file_names


def test_dataset_train_selection_list_error(data_dir_cropped, get_train_file_names):
    with pytest.raises(ValueError):
        CroppedDatasetRGB.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                            data_folder_name=DATA_FOLDER_NAME,
                                            gt_folder_name=GT_FOLDER_NAME,
                                            selection=['something'])


def test_get_gt_data_paths_train(data_dir_cropped, get_train_file_names):
    files_from_method = CroppedDatasetRGB.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                                            data_folder_name=DATA_FOLDER_NAME,
                                                            gt_folder_name=GT_FOLDER_NAME)
    assert len(files_from_method) == 12
    assert files_from_method == get_train_file_names


def test_dataset_rgb_test(dataset_test):
    data_tensor, gt_tensor, idx = dataset_test[0]
    assert data_tensor.shape[-2:] == gt_tensor.shape[-2:]
    assert idx == 0
    assert data_tensor.ndim == 3
    assert gt_tensor.ndim == 3


def test_dataset_rgb_train(dataset_train):
    data_tensor, gt_tensor = dataset_train[0]
    assert data_tensor.shape[-2:] == gt_tensor.shape[-2:]
    assert data_tensor.ndim == 3
    assert gt_tensor.ndim == 3


def test__load_data_and_gt(dataset_train):
    data_img, gt_img = dataset_train._load_data_and_gt(index=0)
    assert data_img.size == gt_img.size
    assert data_img.mode == 'RGB'
    assert gt_img.mode == 'RGB'


@pytest.fixture
def get_train_file_names(data_dir_cropped):
    return [(PosixPath(
        data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0000.png'),
             PosixPath(
                 data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0000_y0000.png'),
             'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0000'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0000_y0150.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0150'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0000_y0187.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0187'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0150_y0000.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0150_y0000'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0150_y0150.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0150_y0150'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0150_y0187.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0150_y0187'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0300_y0000.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0300_y0000'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0300_y0150.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0300_y0150'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0300_y0187.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0300_y0187'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0349_y0000.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0349_y0000'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0349_y0150.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0349_y0150'), (
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/gt/{DATASET_PREFIX}_x0349_y0187.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0349_y0187')]
