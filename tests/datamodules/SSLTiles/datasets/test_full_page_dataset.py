import numpy as np
import pytest
from torchvision.datasets.folder import pil_loader

from src.datamodules.SSLTiles.datasets.full_page_dataset import DatasetSSLTiles
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir

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


def test__cut_image_in_tiles_and_put_together(dataset_train, data_dir, monkeypatch):
    img = data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png'
    img = np.array(pil_loader(img))
    monkeypatch.setattr(dataset_train, 'vertical_shuffle', False)
    monkeypatch.setattr(dataset_train, 'horizontal_shuffle', True)
    np.random.seed(42)
    tile_img, gt = dataset_train._cut_image_in_tiles_and_put_together(img)
    assert False
