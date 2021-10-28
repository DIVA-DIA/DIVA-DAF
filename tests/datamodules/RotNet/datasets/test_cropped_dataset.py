from pathlib import PosixPath

import numpy as np
import pytest
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rotate

from src.datamodules.RotNet.datasets.cropped_dataset import CroppedRotNet, ROTATION_ANGLES
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped


@pytest.fixture
def dataset_train(data_dir_cropped):
    return CroppedRotNet(path=data_dir_cropped / 'train')


def test__load_data_and_gt(dataset_train):
    img = dataset_train._load_data_and_gt(0)
    assert img.size == (300, 300)
    assert img.format == 'PNG'
    assert np.array_equal(np.array(img)[150][150], np.array([97, 72, 32]))


def test__apply_transformation(dataset_train):
    img0_o = dataset_train._load_data_and_gt(0)
    img1_o = dataset_train._load_data_and_gt(1)
    img2_o = dataset_train._load_data_and_gt(2)
    img3_o = dataset_train._load_data_and_gt(3)
    img4_o = dataset_train._load_data_and_gt(4)

    img0, gt0 = dataset_train._apply_transformation(img0_o, 0)
    assert torch.equal(img0, ToTensor()(img0_o))
    assert gt0 == 0

    img1, gt1 = dataset_train._apply_transformation(img1_o, 1)
    assert not torch.equal(ToTensor()(img1_o), img1)
    assert torch.equal(img1, rotate(img=ToTensor()(img1_o), angle=ROTATION_ANGLES[1]))
    assert gt1 == 1

    img2, gt2 = dataset_train._apply_transformation(img2_o, 2)
    assert not torch.equal(ToTensor()(img2_o), img2)
    assert torch.equal(img2, rotate(img=ToTensor()(img2_o), angle=ROTATION_ANGLES[2]))
    assert gt2 == 2

    img3, gt3 = dataset_train._apply_transformation(img3_o, 3)
    assert not torch.equal(ToTensor()(img3_o), img3)
    assert torch.equal(img3, rotate(img=ToTensor()(img3_o), angle=ROTATION_ANGLES[3]))
    assert gt3 == 3

    img4, gt4 = dataset_train._apply_transformation(img4_o, 0)
    assert torch.equal(img4, ToTensor()(img4_o))
    assert gt4 == 0


def test_get_gt_data_paths(data_dir_cropped):
    file_paths = CroppedRotNet.get_gt_data_paths(directory=data_dir_cropped / 'train')
    expected_result = [
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0000_y0000.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0000_y0150.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0000_y0187.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0150_y0000.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0150_y0150.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0150_y0187.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0300_y0000.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0300_y0150.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0300_y0187.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0349_y0000.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0349_y0150.png'),
        PosixPath(
            data_dir_cropped / 'train/data/e-codices_fmb-cb-0055_0098v_max/e-codices_fmb-cb-0055_0098v_max_x0349_y0187.png'),
        ]
    assert len(file_paths) == len(expected_result)
    assert file_paths == expected_result
