import numpy as np
import pytest
from omegaconf import OmegaConf

from src.datamodules.RotNet.datamodule_cropped import RotNetDivaHisDBDataModuleCropped
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

NUM_WORKERS = 4


@pytest.fixture
def data_module_cropped(data_dir_cropped):
    OmegaConf.clear_resolvers()
    return RotNetDivaHisDBDataModuleCropped(data_dir_cropped, num_workers=NUM_WORKERS)


def test_init_datamodule(data_module_cropped):
    assert data_module_cropped.batch_size == 8
    assert data_module_cropped.num_workers == NUM_WORKERS
    assert data_module_cropped.dims == (3, 256, 256)
    assert data_module_cropped.num_classes == 4
    assert np.array_equal(data_module_cropped.class_encodings, [0, 90, 180, 270])
    assert np.array_equal(data_module_cropped.class_weights, [1., 1., 1., 1.])
    assert data_module_cropped.mean == [0.7050454974582426, 0.6503181590413943, 0.5567698583877997]
    assert data_module_cropped.std == [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]
    with pytest.raises(AttributeError):
        getattr(data_module_cropped, 'train')
        getattr(data_module_cropped, 'val')
        getattr(data_module_cropped, 'test')


def test__create_dataset_parameters_cropped(data_module_cropped):
    parameters = data_module_cropped._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert not parameters['is_test']
    assert np.array_equal(parameters['classes'], np.array([0, 90, 180, 270], dtype=np.int64))
