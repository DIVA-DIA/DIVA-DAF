import numpy as np
import pytest
from PIL import ImageChops
from torchvision.datasets.folder import pil_loader

from src.datamodules.SSLTiles.datasets.full_page_dataset import DatasetSSLTiles
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir
from tests.test_data.result_data_ssltiles.result_data import result_img_2_3_horizontal, result_img_2_3_vertical, \
    result_img_2_3_horizontal_vertical

DATA_FOLDER_NAME = 'data'


@pytest.fixture
def dataset_train(data_dir):
    return DatasetSSLTiles(path=data_dir / 'train',
                           data_folder_name=DATA_FOLDER_NAME, gt_folder_name='',
                           image_dims=ImageDimensions(width=960, height=1344),
                           rows=3, cols=2, horizontal_shuffle=True, vertical_shuffle=False)


def test_get_img_gt_path_list(data_dir):
    img_gt_path_list = DatasetSSLTiles.get_img_gt_path_list(directory=data_dir / 'train', data_folder_name='data',
                                                            gt_folder_name='gt',
                                                            selection=None)
    assert len(img_gt_path_list) == 2
    assert img_gt_path_list[0].name == 'fmb-cb-55-005v.png'
    assert img_gt_path_list[1].name == 'fmb-cb-55-005v_2.png'


def test__cut_image_in_tiles_and_put_together_2_3_vertical(dataset_train, data_dir, result_img_2_3_vertical,
                                                           monkeypatch):
    img = data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png'
    img = np.array(pil_loader(img))
    monkeypatch.setattr(dataset_train, 'vertical_shuffle', True)
    monkeypatch.setattr(dataset_train, 'horizontal_shuffle', False)
    np.random.seed(42)
    tile_img, gt = dataset_train._cut_image_in_tiles_and_put_together(img)
    assert gt.shape == (3, 2)
    assert tile_img.width == 960
    assert tile_img.height == 1344
    assert np.array_equal(gt, np.array([[0, 3], [2, 5], [4, 1]]))
    assert ImageChops.difference(tile_img, result_img_2_3_vertical).getbbox() is None


def test__cut_image_in_tiles_and_put_together_2_3_horizontal(dataset_train, data_dir, result_img_2_3_horizontal,
                                                             monkeypatch):
    img = data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png'
    img = np.array(pil_loader(img))
    monkeypatch.setattr(dataset_train, 'vertical_shuffle', False)
    monkeypatch.setattr(dataset_train, 'horizontal_shuffle', True)
    np.random.seed(42)
    tile_img, gt = dataset_train._cut_image_in_tiles_and_put_together(img)
    assert gt.shape == (3, 2)
    assert tile_img.width == 960
    assert tile_img.height == 1344
    assert np.array_equal(gt, np.array([[1, 0], [2, 3], [5, 4]]))
    assert ImageChops.difference(tile_img, result_img_2_3_horizontal).getbbox() is None


def test__cut_image_in_tiles_and_put_together_2_3_horizontal_vertical(dataset_train, data_dir,
                                                                      result_img_2_3_horizontal_vertical, monkeypatch):
    img = data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png'
    img = np.array(pil_loader(img))
    monkeypatch.setattr(dataset_train, 'vertical_shuffle', True)
    monkeypatch.setattr(dataset_train, 'horizontal_shuffle', True)
    np.random.seed(42)
    tile_img, gt = dataset_train._cut_image_in_tiles_and_put_together(img)
    assert gt.shape == (3, 2)
    assert tile_img.width == 960
    assert tile_img.height == 1344
    assert np.array_equal(gt, np.array([[2, 3], [1, 4], [5, 0]]))
    assert ImageChops.difference(tile_img, result_img_2_3_horizontal_vertical).getbbox() is None
