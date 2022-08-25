import pytest

from src.datamodules.SSLTiles.datamodule_prebuilt import SSLTilesDataModulePrebuilt
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir_classification


@pytest.fixture
def get_datamodule(data_dir_classification):
    return SSLTilesDataModulePrebuilt(data_dir=data_dir_classification)


def test_ssltiles_data_module_prebuilt(get_datamodule):
    assert get_datamodule.num_classes == 2
    assert get_datamodule.classes == ['0', '1']


def test__create_dataset_parameters(get_datamodule):
    params = get_datamodule._create_dataset_parameters('train')
    assert 'root' in params
    assert 'transform' in params
    assert params['root'] == get_datamodule.data_dir / 'train'
    assert params['transform'] == get_datamodule.image_transform
