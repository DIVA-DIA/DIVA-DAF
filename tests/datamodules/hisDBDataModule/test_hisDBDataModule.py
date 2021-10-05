import numpy as np
from numpy import uint8
from pytest import fixture

from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCB55, DIVAHisDBDataModuleCropped
from tests.datamodules.hisDBDataModule.dummy_data.dummy_data import data_dir_cropped, data_dir

@fixture
def data_module(data_dir):
    return DIVAHisDBDataModuleCB55(data_dir, num_workers=5)


@fixture
def data_module_cropped(data_dir_cropped):
    return DIVAHisDBDataModuleCropped(data_dir_cropped, num_workers=5)


def test_setup(data_module):
    data_module.setup()
    assert data_module.train is not None
    assert data_module.val is not None


def test_train_dataloader(data_module):
    data_module.setup()
    loader = data_module.train_dataloader()
    assert loader is not None
    assert loader.num_workers == 5
    assert loader.batch_size == 8
    assert loader.dataset is not None


def test_val_dataloader(data_module):
    data_module.setup()
    loader = data_module.val_dataloader()
    assert loader is not None
    assert loader.num_workers == 5
    assert loader.batch_size == 8
    assert loader.dataset is not None


def test_test_dataloader(data_module):
    data_module.setup(stage='test')
    loader = data_module.test_dataloader()
    assert loader is not None
    assert loader.num_workers == 5
    assert loader.batch_size == 8
    assert loader.dataset is not None


def test__create_dataset_parameters(data_module):
    parameters = data_module._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert parameters['workers'] == 5
    assert parameters['imgs_in_memory'] == 4
    assert parameters['crops_per_image'] == 50
    assert parameters['crop_size'] == 256
    assert not parameters['is_test']
    assert np.array_equal(parameters['classes'], np.array([1, 2, 4, 8], dtype=uint8))
    # assert parameters['classes'] == [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]


def test__create_dataset_parameters_cropped(data_module_cropped):
    parameters = data_module_cropped._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert not parameters['is_test']
    assert np.array_equal(parameters['classes'], np.array([1, 2, 4, 8], dtype=uint8))
    # assert parameters['classes'] == [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
