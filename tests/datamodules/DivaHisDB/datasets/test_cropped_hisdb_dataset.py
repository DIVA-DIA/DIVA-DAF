from pathlib import PosixPath

import pytest
import torch

from src.datamodules.DivaHisDB.datasets.cropped_dataset import CroppedHisDBDataset
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

DATA_FOLDER_NAME = 'data'
GT_FOLDER_NAME = 'gt'
DATASET_PREFIX = 'e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max'


@pytest.fixture
def dataset_train(data_dir_cropped):
    return CroppedHisDBDataset(path=data_dir_cropped / f'train', data_folder_name='data', gt_folder_name='gt')


@pytest.fixture
def dataset_val(data_dir_cropped):
    return CroppedHisDBDataset(path=data_dir_cropped / f'val', data_folder_name='data', gt_folder_name='gt')


@pytest.fixture
def dataset_test(data_dir_cropped):
    return CroppedHisDBDataset(path=data_dir_cropped / f'test', data_folder_name='data', gt_folder_name='gt')


def test__load_data_and_gt(dataset_train):
    data_img, gt_img = dataset_train._load_data_and_gt(0)
    assert data_img.size == (300, 300)
    assert gt_img.size == (300, 300)


def test__get_train_val_items_train(dataset_train):
    img, gt, boundary_mask = dataset_train._get_train_val_items(index=0)
    assert img.shape == torch.Size([3, 300, 300])
    assert gt.shape == torch.Size([3, 300, 300])
    assert boundary_mask.shape == torch.Size([300, 300])


def test__get_train_val_items_val(dataset_val):
    img, gt, boundary_mask = dataset_val._get_train_val_items(index=0)
    assert img.shape == torch.Size([3, 300, 300])
    assert gt.shape == torch.Size([3, 300, 300])
    assert boundary_mask.shape == torch.Size([300, 300])


def test__get_train_val_items_test(dataset_test):
    img, gt, boundary_mask, index = dataset_test._get_test_items(index=0)
    assert img.shape == torch.Size([3, 256, 256])
    assert gt.shape == torch.Size([3, 256, 256])
    assert boundary_mask.shape == torch.Size([256, 256])
    assert index == 0


def test_dataset_train_selection_int_error(data_dir_cropped):
    with pytest.raises(ValueError):
        CroppedHisDBDataset.get_gt_data_paths(directory=data_dir_cropped / f'train',
                                              data_folder_name='data', gt_folder_name='gt', selection=2)


def test_dataset_train_selection_int(data_dir_cropped, get_train_file_names):
    files_from_method = CroppedHisDBDataset.get_gt_data_paths(directory=data_dir_cropped / f'train',
                                                              data_folder_name='data', gt_folder_name='gt', selection=1)
    assert len(files_from_method) == 12
    assert files_from_method == get_train_file_names


def test_get_gt_data_paths_train(data_dir_cropped, get_train_file_names):
    files_from_method = CroppedHisDBDataset.get_gt_data_paths(directory=data_dir_cropped / f'train',
                                                              data_folder_name='data', gt_folder_name='gt')
    assert len(files_from_method) == 12
    assert files_from_method == get_train_file_names


def test_get_gt_data_paths_train_selection_string(data_dir_cropped, get_train_file_names):
    files_from_method = CroppedHisDBDataset.get_gt_data_paths(directory=data_dir_cropped / f'train',
                                                              data_folder_name='data', gt_folder_name='gt',
                                                              selection=['e-codices_fmb-cb-0055_0098v_max'])
    assert files_from_method == get_train_file_names
    assert len(files_from_method) == 12


def test_get_gt_data_paths_val_selection_negative_int(data_dir_cropped):
    with pytest.raises(ValueError):
        CroppedHisDBDataset.get_gt_data_paths(directory=data_dir_cropped / f'val',
                                              data_folder_name='data', gt_folder_name='gt',
                                              selection=-1)


def test_get_gt_data_paths_val(data_dir_cropped):
    files_from_method = CroppedHisDBDataset.get_gt_data_paths(directory=data_dir_cropped / f'val',
                                                              data_folder_name='data', gt_folder_name='gt')
    expected_result = [
        (PosixPath(
            data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0000_y0000.png'),
         PosixPath(
             data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0000_y0000.png'),
         'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0000'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0000_y0150.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0000_y0150.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0150'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0000_y0187.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0000_y0187.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0187'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0150_y0000.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0150_y0000.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0150_y0000'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0150_y0150.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0150_y0150.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0150_y0150'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0150_y0187.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0150_y0187.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0150_y0187'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0300_y0000.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0300_y0000.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0300_y0000'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0300_y0150.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0300_y0150.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0300_y0150'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0300_y0187.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0300_y0187.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0300_y0187'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0349_y0000.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0349_y0000.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0349_y0000'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0349_y0150.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0349_y0150.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0349_y0150'), (
            PosixPath(
                data_dir_cropped / f'val/data/{DATASET_PREFIX}_x0349_y0187.png'),
            PosixPath(
                data_dir_cropped / f'val/gt/{DATASET_PREFIX}_x0349_y0187.png'),
            'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0349_y0187')]
    assert len(files_from_method) == 12
    assert files_from_method == expected_result


def test_get_gt_data_paths_test(data_dir_cropped):
    files_from_method = CroppedHisDBDataset.get_gt_data_paths(directory=data_dir_cropped / f'test',
                                                              data_folder_name='data', gt_folder_name='gt')
    expected_result = [(PosixPath(
        data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0000_y0000.png'),
                        PosixPath(
                            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0000_y0000.png'),
                        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0000'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0000_y0128.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0000_y0128.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0128'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0000_y0231.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0000_y0231.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0231'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0128_y0000.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0128_y0000.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0128_y0000'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0128_y0128.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0128_y0128.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0128_y0128'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0128_y0231.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0128_y0231.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0128_y0231'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0256_y0000.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0256_y0000.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0256_y0000'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0256_y0128.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0256_y0128.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0256_y0128'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0256_y0231.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0256_y0231.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0256_y0231'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0384_y0000.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0384_y0000.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0384_y0000'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0384_y0128.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0384_y0128.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0384_y0128'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0384_y0231.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0384_y0231.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0384_y0231'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0393_y0000.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0393_y0000.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0393_y0000'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0393_y0128.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0393_y0128.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0393_y0128'), (
        PosixPath(
            data_dir_cropped / f'test/data/{DATASET_PREFIX}_x0393_y0231.png'),
        PosixPath(
            data_dir_cropped / f'test/gt/{DATASET_PREFIX}_x0393_y0231.png'),
        'e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0393_y0231')]
    assert len(files_from_method) == 15
    assert files_from_method == expected_result


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
