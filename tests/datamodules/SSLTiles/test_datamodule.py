import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.datamodules.SSLTiles.datamodule import SSLTilesDataModule
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir

NUM_WORKERS = 4
DATA_FOLDER_NAME = 'data'


@pytest.fixture
def data_module_ssltiles(data_dir):
    OmegaConf.clear_resolvers()
    return SSLTilesDataModule(data_dir=data_dir,
                              data_folder_name=DATA_FOLDER_NAME,
                              rows=3, cols=2, horizontal_shuffle=True, vertical_shuffle=True,
                              num_workers=NUM_WORKERS)


def test_init_datmodule(data_module_ssltiles):
    assert data_module_ssltiles.batch_size == 8
    assert data_module_ssltiles.num_workers == NUM_WORKERS
    assert data_module_ssltiles.dims == (3, 960, 1344)
    assert data_module_ssltiles.num_classes == 6
    assert np.array_equal(data_module_ssltiles.class_encodings, [0, 1, 2, 3, 4, 5])
    assert torch.allclose(data_module_ssltiles.class_weights, torch.tensor([.16, .16, .16, .16, .16, .16]), atol=1e-02)
    assert data_module_ssltiles.mean == [0.8092701646630874, 0.7459951880057579, 0.6299246773362123]
    assert data_module_ssltiles.std == [0.25536060627463697, 0.24617490612180548, 0.22308898058230786]
    with pytest.raises(AttributeError):
        getattr(data_module_ssltiles, 'train')
        getattr(data_module_ssltiles, 'val')
        getattr(data_module_ssltiles, 'test')


def test__create_dataset_parameters(data_module_ssltiles):
    parameters = data_module_ssltiles._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert 'is_test' is not parameters
    assert parameters['rows'] == 3
    assert parameters['cols'] == 2
    assert parameters['horizontal_shuffle']
    assert parameters['vertical_shuffle']
