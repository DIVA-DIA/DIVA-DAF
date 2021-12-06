import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from src.datamodules.RGB.datamodule_cropped import DataModuleCroppedRGB
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped
from tests.datamodules.DivaHisDB.datasets.test_cropped_hisdb_dataset import dataset_test

NUM_WORKERS = 4


@pytest.fixture
def data_module_cropped_rgb(data_dir_cropped):
    OmegaConf.clear_resolvers()
    datamodules = DataModuleCroppedRGB(data_dir_cropped, data_folder_name='data', gt_folder_name='gt',
                                       num_workers=NUM_WORKERS)
    return datamodules


@pytest.fixture()
def class_encodings():
    return [(0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 0, 8), (128, 0, 1), (128, 0, 2),
            (128, 0, 4), (128, 0, 8)]


def test_setup_fit(data_module_cropped_rgb, monkeypatch):
    stage = 'fit'
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_cropped_rgb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_rgb)
    data_module_cropped_rgb.setup(stage)
    assert not hasattr(data_module_cropped_rgb, 'test')
    assert hasattr(data_module_cropped_rgb, 'train')
    assert hasattr(data_module_cropped_rgb, 'val')


def test_setup_test(data_module_cropped_rgb, monkeypatch):
    stage = 'test'
    trainer = Trainer(accelerator='ddp')
    monkeypatch.setattr(data_module_cropped_rgb, 'trainer', trainer)
    monkeypatch.setattr(trainer, 'datamodule', data_module_cropped_rgb)
    data_module_cropped_rgb.setup(stage)
    assert hasattr(data_module_cropped_rgb, 'test')
    assert not hasattr(data_module_cropped_rgb, 'train')
    assert not hasattr(data_module_cropped_rgb, 'val')


def test_init_datamodule(data_module_cropped_rgb, class_encodings):
    assert data_module_cropped_rgb.batch_size == 8
    assert data_module_cropped_rgb.num_workers == NUM_WORKERS
    assert data_module_cropped_rgb.dims == (3, 256, 256)
    assert data_module_cropped_rgb.num_classes == 8
    assert data_module_cropped_rgb.class_encodings == class_encodings
    assert torch.equal(data_module_cropped_rgb.class_weights,
                       torch.tensor([0.00047088927, 0.011501364, 0.1453358, 0.003524866, 0.6084134, 0.018242653,
                                     0.20573607, 0.0067749745]))
    assert data_module_cropped_rgb.mean == [0.7050454974582426, 0.6503181590413943, 0.5567698583877997]
    assert data_module_cropped_rgb.std == [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]
    with pytest.raises(AttributeError):
        getattr(data_module_cropped_rgb, 'train')
        getattr(data_module_cropped_rgb, 'val')
        getattr(data_module_cropped_rgb, 'test')


def test_get_img_name_coordinates(data_module_cropped_rgb, dataset_test):
    data_module_cropped_rgb.test = dataset_test
    coords_0 = data_module_cropped_rgb.get_img_name_coordinates(0)
    coords_1 = data_module_cropped_rgb.get_img_name_coordinates(1)
    coords_2 = data_module_cropped_rgb.get_img_name_coordinates(2)
    assert coords_0 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0000')
    assert coords_1 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0128')
    assert coords_2 == ('e-codices_fmb-cb-0055_0098v_max', 'e-codices_fmb-cb-0055_0098v_max_x0000_y0231')


def test__create_dataset_parameters_cropped(data_module_cropped_rgb, class_encodings):
    parameters = data_module_cropped_rgb._create_dataset_parameters()
    assert 'train' in str(parameters['path'])
    assert not parameters['is_test']
    assert np.array_equal(parameters['classes'], np.array(class_encodings, dtype=np.uint8))
