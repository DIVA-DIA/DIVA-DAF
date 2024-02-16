import numpy as np
import pytest
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor

from src.datamodules.SSLTiles.utils.single_transform import TilesBuilding

from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir


@pytest.fixture()
def pil_image(data_dir):
    return pil_loader(data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png')


def test_tiles_transformation_fail_too_many_fixed_positions(data_dir):
    with pytest.raises(ValueError):
        _ = TilesBuilding(rows=2, cols=2, fixed_positions=3, width_center_crop=840, height_center_crop=1200)


def test_tiles_transformation_fail_fixed_positions_too_big(data_dir):
    with pytest.raises(ValueError):
        _ = TilesBuilding(rows=2, cols=2, fixed_positions=5, width_center_crop=840, height_center_crop=1200)


def test_tiles_transformation(pil_image):
    img_tensor = ToTensor()(pil_image)
    transform = TilesBuilding(rows=3, cols=2, fixed_positions=3, width_center_crop=840, height_center_crop=1200)
    assert transform(img_tensor).shape == img_tensor.shape
    assert transform.rows == 3
    assert transform.cols == 2
    assert transform.permutations.shape == (720, 6)
    assert transform.classes.shape == (6,)
    assert transform.fixed_positions == 3
    assert transform.filtered_perms.shape == (40, 6)
    assert transform.width_center_crop == 840
    assert transform.height_center_crop == 1200
    assert transform.width_tile == 420
    assert transform.height_tile == 400
    assert transform.target_class is not None


def test_tiles_transformation_call(mocker, pil_image):
    img_tensor = ToTensor()(pil_image)
    transform = TilesBuilding(rows=3, cols=2, fixed_positions=3, width_center_crop=840, height_center_crop=1200)
    mocker.patch('numpy.random.randint', return_value=0)
    transform(img_tensor)
    assert transform.target_class == 0


def test__update_target_perm():
    transform = TilesBuilding(rows=3, cols=2, fixed_positions=3, width_center_crop=840, height_center_crop=1200)
    assert transform.target_class is None
    transform._update_target_perm()
    assert transform.target_class is not None


def test__get_perms_with_n_fixed_positions():
    transform = TilesBuilding(rows=2, cols=2, fixed_positions=2, width_center_crop=840, height_center_crop=1200)
    assert np.all(np.equal(transform._get_perms_with_n_fixed_positions(), np.array(
        [[0, 1, 3, 2], [0, 2, 1, 3], [0, 3, 2, 1], [1, 0, 2, 3], [2, 1, 0, 3], [3, 1, 2, 0]])))


def test__get_tile_image(pil_image):
    transform = TilesBuilding(rows=2, cols=2, fixed_positions=2, width_center_crop=840, height_center_crop=1200)
    img_tensor = ToTensor()(pil_image)
    trans_ig = transform(img_tensor)
    assert trans_ig.shape == img_tensor.shape
