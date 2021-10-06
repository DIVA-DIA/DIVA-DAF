import numpy as np
from numpy import uint8
from pytest import fixture

from src.datamodules.hisDBDataModule.DIVAHisDBDataModule import DIVAHisDBDataModuleCropped
from tests.datamodules.hisDBDataModule.dummy_data.dummy_data import data_dir_cropped, data_dir


@fixture
def data_module_cropped(data_dir_cropped):
    return DIVAHisDBDataModuleCropped(data_dir_cropped, num_workers=5)


def test__create_dataset_parameters_cropped(data_module_cropped):
    parameters = data_module_cropped._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert not parameters['is_test']
    assert np.array_equal(parameters['classes'], np.array([1, 2, 4, 8], dtype=uint8))
    # assert parameters['classes'] == [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
