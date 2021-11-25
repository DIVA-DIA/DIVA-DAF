import pytest

from src.datamodules.RGB.datasets.full_page_dataset import DatasetRGB, ImageDimensions
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir


@pytest.fixture
def dataset_train(data_dir):
    return DatasetRGB(path=data_dir / 'train', data_folder_name='data', gt_folder_name='gt',
                      image_dims=ImageDimensions(width=487, height=649))


def test_get_gt_data_paths(data_dir):
    file_list = DatasetRGB.get_img_gt_path_list(directory=data_dir / 'train', data_folder_name='data', gt_folder_name='gt')
    assert len(file_list) == 1
    assert file_list[0] == (data_dir / 'train' / 'data' / 'e-codices_fmb-cb-0055_0098v_max.jpg',
                            data_dir / 'train' / 'gt' / 'e-codices_fmb-cb-0055_0098v_max.png',
                            'e-codices_fmb-cb-0055_0098v_max')


def test_dataset_rgb(dataset_train):
    data_tensor, gt_tensor = dataset_train[0]
    assert data_tensor.shape[-2:] == gt_tensor.shape[-2:]
    assert data_tensor.ndim == 3
    assert gt_tensor.ndim == 3


def test__load_data_and_gt(dataset_train):
    data_img, gt_img = dataset_train._load_data_and_gt(index=0)
    assert data_img.size == gt_img.size
    assert data_img.mode == 'RGB'
    assert gt_img.mode == 'RGB'
