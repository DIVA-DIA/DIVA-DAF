import numpy as np
import pytest
import torch
from PIL import ImageChops
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor

from src.datamodules.SSLTiles.datasets.full_page_dataset import DatasetSSLTiles
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir
from tests.test_data.result_data_ssltiles.result_data import result_img_2_3_horizontal, result_img_2_3_vertical, \
    result_img_2_3_horizontal_vertical

DATA_FOLDER_NAME = 'data'


@pytest.fixture
def dataset_train_ssl(data_dir):
    return DatasetSSLTiles(path=data_dir / 'train',
                           data_folder_name=DATA_FOLDER_NAME, gt_folder_name='',
                           image_dims=ImageDimensions(width=960, height=1344),
                           rows=3, cols=2, horizontal_shuffle=True, vertical_shuffle=False)



def test__cut_image_in_tiles_and_put_together_2_3_vertical(dataset_train_ssl, data_dir, result_img_2_3_vertical,
                                                           monkeypatch):
    img_path = data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png'
    img = ToTensor()(pil_loader(img_path))
    monkeypatch.setattr(dataset_train_ssl, 'vertical_shuffle', True)
    monkeypatch.setattr(dataset_train_ssl, 'horizontal_shuffle', False)
    np.random.seed(42)
    tile_img, gt = dataset_train_ssl._cut_image_in_tiles_and_put_together(img)
    assert gt.shape == (3, 2)
    assert tile_img.width == 960
    assert tile_img.height == 1344
    assert np.array_equal(gt, np.array([[0, 3], [2, 5], [4, 1]]))
    assert ImageChops.difference(tile_img, result_img_2_3_vertical).getbbox() is None


def test__cut_image_in_tiles_and_put_together_2_3_horizontal(dataset_train_ssl, data_dir, result_img_2_3_horizontal,
                                                             monkeypatch):
    img_path = data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png'
    img = ToTensor()(pil_loader(img_path))
    monkeypatch.setattr(dataset_train_ssl, 'vertical_shuffle', False)
    monkeypatch.setattr(dataset_train_ssl, 'horizontal_shuffle', True)
    np.random.seed(42)
    tile_img, gt = dataset_train_ssl._cut_image_in_tiles_and_put_together(img)
    assert gt.shape == (3, 2)
    assert tile_img.width == 960
    assert tile_img.height == 1344
    assert np.array_equal(gt, np.array([[1, 0], [2, 3], [5, 4]]))
    assert ImageChops.difference(tile_img, result_img_2_3_horizontal).getbbox() is None


def test__cut_image_in_tiles_and_put_together_2_3_horizontal_vertical(dataset_train_ssl, data_dir,
                                                                      result_img_2_3_horizontal_vertical, monkeypatch):
    img_path = data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png'
    img_array = ToTensor()(pil_loader(img_path))
    monkeypatch.setattr(dataset_train_ssl, 'vertical_shuffle', True)
    monkeypatch.setattr(dataset_train_ssl, 'horizontal_shuffle', True)
    np.random.seed(42)
    tile_img, gt = dataset_train_ssl._cut_image_in_tiles_and_put_together(img_array)
    assert gt.shape == (3, 2)
    assert tile_img.width == 960
    assert tile_img.height == 1344
    assert np.array_equal(gt, np.array([[2, 3], [1, 4], [5, 0]]))
    assert ImageChops.difference(tile_img, result_img_2_3_horizontal_vertical).getbbox() is None


def test__load_data_and_gt(dataset_train_ssl):
    img = dataset_train_ssl._load_data_and_gt(0)
    assert img.size == (960, 1344)
    assert np.array_equal(np.array(img)[150][150], np.array([236, 225, 199]))


def test__apply_transformation(dataset_train_ssl):
    org0 = dataset_train_ssl._load_data_and_gt(0)
    org1 = dataset_train_ssl._load_data_and_gt(1)

    np.random.seed(42)
    img0, gt0 = dataset_train_ssl._apply_transformation(org0, 0)
    np.random.seed(42)
    img_index0, gt_index0 = dataset_train_ssl[0]
    assert torch.equal(img0, img_index0)
    assert torch.equal(gt0, gt_index0)

    np.random.seed(42)
    img1, gt1 = dataset_train_ssl._apply_transformation(org0, 1)
    np.random.seed(42)
    img_index1, gt_index1 = dataset_train_ssl[1]
    assert not torch.equal(ToTensor()(org0), img1)
    assert torch.equal(img1, img_index1)
    assert torch.equal(gt1, gt_index1)

    np.random.seed(42)
    img2, gt2 = dataset_train_ssl._apply_transformation(org1, 0)
    np.random.seed(42)
    img_index2, gt_index2 = dataset_train_ssl[0]
    assert torch.equal(img2, img_index2)
    assert torch.equal(gt2, gt_index2)

    np.random.seed(42)
    img3, gt3 = dataset_train_ssl._apply_transformation(org1, 1)
    np.random.seed(42)
    img_index3, gt_index3 = dataset_train_ssl[1]
    assert not torch.equal(ToTensor()(org1), img2)
    assert torch.equal(img3, img_index3)
    assert torch.equal(gt3, gt_index3)


def test_get_img_gt_path_list(data_dir):
    img_gt_path_list = DatasetSSLTiles.get_img_gt_path_list(directory=data_dir / 'train', data_folder_name='data',
                                                            gt_folder_name='gt',
                                                            selection=None)
    assert len(img_gt_path_list) == 2
    assert img_gt_path_list[0].name == 'fmb-cb-55-005v.png'
    assert img_gt_path_list[1].name == 'fmb-cb-55-005v_2.png'


def test_get_gt_data_paths_selection_int_negative(data_dir):
    with pytest.raises(ValueError):
        DatasetSSLTiles.get_img_gt_path_list(directory=data_dir / 'train',
                                             data_folder_name=DATA_FOLDER_NAME, selection=-1)


def test_get_gt_data_paths_selection_int_too_big(data_dir):
    with pytest.raises(ValueError):
        DatasetSSLTiles.get_img_gt_path_list(directory=data_dir / 'train',
                                             data_folder_name=DATA_FOLDER_NAME, selection=3)


def test_get_gt_data_paths_selection_list(data_dir):
    file_paths = DatasetSSLTiles.get_img_gt_path_list(directory=data_dir / 'train',
                                                      data_folder_name=DATA_FOLDER_NAME,
                                                      selection=['fmb-cb-55-005v'])

    assert len(file_paths) == 1
    assert file_paths == [data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png']
