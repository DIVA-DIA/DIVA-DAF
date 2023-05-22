import pytest
import torch
from omegaconf import OmegaConf

from src.datamodules.DivaHisDB.datamodule_cropped import DivaHisDBDataModuleCropped
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped
from tests.datamodules.DivaHisDB.datasets.test_cropped_hisdb_dataset import dataset_test

NUM_WORKERS = 4


@pytest.fixture
def data_module_cropped_hisdb(data_dir_cropped):
    OmegaConf.clear_resolvers()
    datamodules = DivaHisDBDataModuleCropped(data_dir_cropped, data_folder_name='data', gt_folder_name='gt',
                                             num_workers=NUM_WORKERS)
    return datamodules


def test_init_datamodule(data_module_cropped_hisdb):
    assert data_module_cropped_hisdb.batch_size == 8
    assert data_module_cropped_hisdb.num_workers == NUM_WORKERS
    assert data_module_cropped_hisdb.dims == (3, 256, 256)
    assert data_module_cropped_hisdb.num_classes == 4
    assert data_module_cropped_hisdb.class_encodings == [1, 2, 4, 8]
    assert torch.equal(data_module_cropped_hisdb.class_weights,
                       torch.tensor(
                           [0.004952207651647859, 0.07424270397485577, 0.8964025044572563, 0.02440258391624002]))
    assert data_module_cropped_hisdb.mean == [0.7050454974582426, 0.6503181590413943, 0.5567698583877997]
    assert data_module_cropped_hisdb.std == [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]
    with pytest.raises(AttributeError):
        getattr(data_module_cropped_hisdb, 'train')
        getattr(data_module_cropped_hisdb, 'val')
        getattr(data_module_cropped_hisdb, 'test')


def test_get_img_name_coordinates(data_module_cropped_hisdb, dataset_test):
    data_module_cropped_hisdb.test = dataset_test
    coords_0 = data_module_cropped_hisdb.get_img_name_coordinates(0)
    coords_1 = data_module_cropped_hisdb.get_img_name_coordinates(1)
    coords_2 = data_module_cropped_hisdb.get_img_name_coordinates(2)
    assert coords_0 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0000')
    assert coords_1 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0128')
    assert coords_2 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0231')


def test__create_dataset_parameters_cropped(data_module_cropped_hisdb):
    parameters = data_module_cropped_hisdb._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert not parameters['is_test']
