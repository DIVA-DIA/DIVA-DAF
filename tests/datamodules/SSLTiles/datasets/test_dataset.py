import numpy as np
import pytest
import torch
from PIL import ImageChops
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor

from src.datamodules.SSLTiles.datasets.dataset import DatasetSSLTiles
from src.datamodules.SSLTiles.utils.misc import GTType
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir
from tests.test_data.result_data_ssltiles.result_data import result_img_2_3_horizontal, result_img_2_3_vertical, \
    result_img_2_3_horizontal_vertical

DATA_FOLDER_NAME = 'data'


@pytest.fixture
def dataset_train_ssl(data_dir):
    return DatasetSSLTiles(path=data_dir / 'train',
                           data_folder_name=DATA_FOLDER_NAME,
                           gt_type=GTType.ROW_COLUMN,
                           image_dims=ImageDimensions(width=960, height=1344),
                           rows=3, cols=2, horizontal_shuffle=True, vertical_shuffle=False)


def test__load_data_and_gt(dataset_train_ssl):
    img = dataset_train_ssl._load_data_and_gt(0)
    assert img.size == (960, 1344)
    assert np.array_equal(np.array(img)[150][150], np.array([236, 225, 199]))


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
