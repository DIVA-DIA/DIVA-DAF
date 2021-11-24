import numpy as np
import pytest
import torch
from numpy import uint8
from omegaconf import OmegaConf

from src.datamodules.DivaHisDB.datamodule_cropped import DivaHisDBDataModuleCropped
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped
from tests.datamodules.DivaHisDB.datasets.test_cropped_hisdb_dataset import dataset_test

NUM_WORKERS = 4


@pytest.fixture
def data_module_cropped(data_dir_cropped):
    OmegaConf.clear_resolvers()
    datamodules = DivaHisDBDataModuleCropped(data_dir_cropped, data_folder_name='data', gt_folder_name='gt',
                                             num_workers=NUM_WORKERS)
    return datamodules


def test_init_datamodule(data_module_cropped):
    assert data_module_cropped.batch_size == 8
    assert data_module_cropped.num_workers == NUM_WORKERS
    assert data_module_cropped.dims == (3, 256, 256)
    assert data_module_cropped.num_classes == 4
    assert data_module_cropped.class_encodings == [1, 2, 4, 8]
    assert torch.all(
        torch.eq(data_module_cropped.class_weights,
                 torch.tensor([0.004952207651647859, 0.07424270397485577, 0.8964025044572563, 0.02440258391624002])))
    assert data_module_cropped.mean == [0.7050454974582426, 0.6503181590413943, 0.5567698583877997]
    assert data_module_cropped.std == [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]
    with pytest.raises(AttributeError):
        getattr(data_module_cropped, 'train')
        getattr(data_module_cropped, 'val')
        getattr(data_module_cropped, 'test')


def test_get_img_name_coordinates(data_module_cropped, dataset_test):
    data_module_cropped.test = dataset_test
    coords_0 = data_module_cropped.get_img_name_coordinates(0)
    coords_1 = data_module_cropped.get_img_name_coordinates(1)
    coords_2 = data_module_cropped.get_img_name_coordinates(2)
    assert coords_0 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0000', (0, 0))
    assert coords_1 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0128', (0, 128))
    assert coords_2 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0231', (0, 231))


def test__create_dataset_parameters_cropped(data_module_cropped):
    parameters = data_module_cropped._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert not parameters['is_test']
    assert np.array_equal(parameters['classes'], np.array([1, 2, 4, 8], dtype=uint8))
