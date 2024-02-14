import pytest
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import ToTensor

from src.datamodules.SSLTiles.utils.single_transform import TilesBuilding

from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir


def test_tiles_transformation_fail_check(data_dir):
    with pytest.raises(ValueError):
        _ = TilesBuilding(rows=2, cols=2, fixed_positions=3, width_center_crop=840, height_center_crop=1200)


def test_tiles_transformation(data_dir):
    img = pil_loader(data_dir / 'train' / 'data' / 'fmb-cb-55-005v.png')
    img_tensor = ToTensor()(img)
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


@pytest.mark.skip('under construction')
def test__update_target_perm():
    assert False


@pytest.mark.skip('under construction')
def test__get_perms_with_n_fixed_positions():
    assert False


@pytest.mark.skip('under construction')
def test__get_tile_image():
    assert False
