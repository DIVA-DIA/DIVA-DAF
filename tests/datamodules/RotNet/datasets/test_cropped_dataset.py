from pathlib import PosixPath

import numpy as np
import pytest
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rotate

from src.datamodules.RotNet.datasets.cropped_dataset import CroppedRotNet
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

DATA_FOLDER_NAME = 'data'
GT_FOLDER_NAME = None
DATASET_PREFIX = 'e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max'


@pytest.fixture
def dataset_train(data_dir_cropped):
    return CroppedRotNet(path=data_dir_cropped / 'train',
                         data_folder_name=DATA_FOLDER_NAME)


def test__load_data_and_gt(dataset_train):
    img = dataset_train._load_data_and_gt(0)
    assert img.size == (300, 300)
    assert np.array_equal(np.array(img)[150][150], np.array([97, 72, 32]))


def test__apply_transformation(dataset_train, mocker):
    org0 = dataset_train._load_data_and_gt(0)

    mocker.patch('torch.randint', return_value=torch.tensor([0]))
    img0, gt0 = dataset_train._apply_transformation(org0, -1)
    img_index0, gt_index0 = dataset_train[0]
    assert torch.equal(img0, ToTensor()(org0))
    assert torch.equal(img0, img_index0)
    assert gt0 == gt_index0
    assert gt0 == 0

    mocker.patch('torch.randint', return_value=torch.tensor([1]))
    img1, gt1 = dataset_train._apply_transformation(org0, -1)
    img_index1, gt_index1 = dataset_train[0]
    assert not torch.equal(ToTensor()(org0), img1)
    assert torch.equal(img1, rotate(img=ToTensor()(org0), angle=90))
    assert torch.equal(img1, img_index1)
    assert gt1 == gt_index1
    assert gt1 == 1

    mocker.patch('torch.randint', return_value=torch.tensor([2]))
    img2, gt2 = dataset_train._apply_transformation(org0, -1)
    img_index2, gt_index2 = dataset_train[0]
    assert not torch.equal(ToTensor()(org0), img2)
    assert torch.equal(img2, rotate(img=ToTensor()(org0), angle=180))
    assert torch.equal(img2, img_index2)
    assert gt2 == gt_index2
    assert gt2 == 2

    mocker.patch('torch.randint', return_value=torch.tensor([3]))
    img3, gt3 = dataset_train._apply_transformation(org0, -1)
    img_index3, gt_index3 = dataset_train[0]
    assert not torch.equal(ToTensor()(org0), img3)
    assert torch.equal(img3, rotate(img=ToTensor()(org0), angle=270))
    assert torch.equal(img3, img_index3)
    assert gt3 == gt_index3
    assert gt3 == 3


def test_get_gt_data_paths(data_dir_cropped):
    file_paths = CroppedRotNet.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                                 data_folder_name=DATA_FOLDER_NAME)

    expected_result = [
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0187.png'),
    ]
    assert len(file_paths) == len(expected_result)
    assert file_paths == expected_result


def test_get_gt_data_paths_selection_int(data_dir_cropped):
    file_paths = CroppedRotNet.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                                 data_folder_name=DATA_FOLDER_NAME, selection=1)

    expected_result = [
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0187.png'),
    ]
    assert len(file_paths) == len(expected_result)
    assert file_paths == expected_result


def test_get_gt_data_paths_selection_int_negative(data_dir_cropped):
    with pytest.raises(ValueError):
        CroppedRotNet.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                        data_folder_name=DATA_FOLDER_NAME, selection=-1)


def test_get_gt_data_paths_selection_int_too_big(data_dir_cropped):
    with pytest.raises(ValueError):
        CroppedRotNet.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                        data_folder_name=DATA_FOLDER_NAME, selection=3)


def test_get_gt_data_paths_selection_list(data_dir_cropped):
    file_paths = CroppedRotNet.get_gt_data_paths(directory=data_dir_cropped / 'train',
                                                 data_folder_name=DATA_FOLDER_NAME,
                                                 selection=['e-codices_fmb-cb-0055_0098v_max'])

    expected_result = [
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0000_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0150_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0300_y0187.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0000.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0150.png'),
        PosixPath(
            data_dir_cropped / f'train/data/{DATASET_PREFIX}_x0349_y0187.png'),
    ]
    assert len(file_paths) == len(expected_result)
    assert file_paths == expected_result
