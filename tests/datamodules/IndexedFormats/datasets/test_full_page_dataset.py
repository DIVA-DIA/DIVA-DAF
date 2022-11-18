import pytest
from torch import is_tensor

from src.datamodules.IndexedFormats.datasets.full_page_dataset import DatasetIndexed
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_fixed_gif.dummy_data import data_dir


@pytest.fixture
def dataset_train(data_dir):
    return DatasetIndexed(path=data_dir / 'train', data_folder_name='data', gt_folder_name='gt',
                          image_dims=ImageDimensions(width=960, height=1344))


@pytest.fixture
def dataset_val(data_dir):
    return DatasetIndexed(path=data_dir / 'val', data_folder_name='data', gt_folder_name='gt',
                          image_dims=ImageDimensions(width=960, height=1344))


@pytest.fixture
def dataset_test(data_dir):
    return DatasetIndexed(path=data_dir / 'test', data_folder_name='data', gt_folder_name='gt',
                          image_dims=ImageDimensions(width=960, height=1344), is_test=True)


def test_init_train(dataset_train):
    assert not dataset_train.is_test
    assert dataset_train.num_samples == 2
    assert len(dataset_train.img_gt_path_list) == 2
    assert len(dataset_train.img_gt_path_list[0]) == 2
    assert not hasattr(dataset_test, 'output_file_list')


def test_init_val(dataset_val):
    assert not dataset_val.is_test
    assert dataset_val.num_samples == 2
    assert len(dataset_val.img_gt_path_list) == 2
    assert len(dataset_val.img_gt_path_list[0]) == 2
    assert not hasattr(dataset_test, 'output_file_list')


def test_init_test(dataset_test):
    assert dataset_test.is_test
    assert dataset_test.num_samples == 2
    assert len(dataset_test.img_gt_path_list) == 2
    assert len(dataset_test.img_gt_path_list[0]) == 2
    assert hasattr(dataset_test, 'output_file_list')


def test__get_item(dataset_train):
    train_tuple = dataset_train[0]
    assert len(train_tuple) == 2
    assert is_tensor(train_tuple[0])
    assert train_tuple[0].shape == (3, 1344, 960)
    assert is_tensor(train_tuple[1])
    assert train_tuple[1].shape == (1344, 960)


def test_get_img_gt_path_list(data_dir):
    paths = DatasetIndexed.get_img_gt_path_list(directory=data_dir / 'train', data_folder_name='data',
                                                gt_folder_name='gt')
    expected_paths = [
        (data_dir / 'train' / 'data' / '2022C-01-dum-folioN-1000.jpg', data_dir / 'train' / 'gt' / '2022C-01-dum-gtL-1000.gif'),
        (data_dir / 'train' / 'data' / '2022C-01-dum-folioN-1001.jpg', data_dir / 'train' / 'gt' / '2022C-01-dum-gtL-1001.gif')]
    assert paths == expected_paths
