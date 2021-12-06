import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.datamodules.RotNet.datamodule_cropped import RotNetDivaHisDBDataModuleCropped
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

NUM_WORKERS = 4
DATA_FOLDER_NAME = 'data'

@pytest.fixture
def data_module_cropped_rotnet(data_dir_cropped):
    OmegaConf.clear_resolvers()
    return RotNetDivaHisDBDataModuleCropped(data_dir=data_dir_cropped,
                                            data_folder_name=DATA_FOLDER_NAME,
                                            num_workers=NUM_WORKERS)


def test_init_datamodule(data_module_cropped_rotnet):
    assert data_module_cropped_rotnet.batch_size == 8
    assert data_module_cropped_rotnet.num_workers == NUM_WORKERS
    assert data_module_cropped_rotnet.dims == (3, 256, 256)
    assert data_module_cropped_rotnet.num_classes == 4
    assert np.array_equal(data_module_cropped_rotnet.class_encodings, [0, 90, 180, 270])
    assert torch.equal(data_module_cropped_rotnet.class_weights, torch.tensor([.25, .25, .25, .25]))
    assert data_module_cropped_rotnet.mean == [0.7050454974582426, 0.6503181590413943, 0.5567698583877997]
    assert data_module_cropped_rotnet.std == [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]
    with pytest.raises(AttributeError):
        getattr(data_module_cropped_rotnet, 'train')
        getattr(data_module_cropped_rotnet, 'val')
        getattr(data_module_cropped_rotnet, 'test')


def test__create_dataset_parameters_cropped(data_module_cropped_rotnet):
    parameters = data_module_cropped_rotnet._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert not parameters['is_test']
    assert np.array_equal(parameters['classes'], np.array([0, 90, 180, 270], dtype=np.int64))
