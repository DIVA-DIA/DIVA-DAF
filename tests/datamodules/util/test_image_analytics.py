import numpy as np

from src.datamodules.utils.image_analytics import compute_mean_std, _return_mean, _return_std
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped, data_dir


def test_compute_mean_std_inmem(data_dir_cropped):
    path_to_files = data_dir_cropped / 'train' / 'data' / 'e-codices_fmb-cb-0055_0098v_max'
    path_list = list(path_to_files.iterdir())
    mean, std = compute_mean_std(file_names=path_list, inmem=True, workers=1)
    assert np.isclose(mean, [0.7050454974582425, 0.6503181590413943, 0.5567698583877996]).any()
    assert np.isclose(std, [0.31040608596198827, 0.30533118388840325, 0.28919611393432737]).any()


def test_compute_mean_std_not_inmem(data_dir_cropped):
    path_to_files = data_dir_cropped / 'train' / 'data' / 'e-codices_fmb-cb-0055_0098v_max'
    path_list = list(path_to_files.iterdir())
    mean, std = compute_mean_std(file_names=path_list, inmem=False, workers=1)
    assert np.isclose(mean, [0.7050454974582426, 0.6503181590413943, 0.5567698583877997]).any()
    assert np.isclose(std, [0.3104060859619883, 0.30533118388840325, 0.28919611393432726]).any()


def test__return_mean(data_dir):
    path_to_files = data_dir / 'train' / 'data'
    path_file = list(path_to_files.iterdir())[0]
    mean = _return_mean(image_path=path_file)
    assert np.allclose(mean, [0.6613600924561268, 0.6080705925283078, 0.5188177611400755], rtol=2e-02)


def test__return_std(data_dir):
    path_to_files = data_dir / 'train' / 'data'
    path_file = list(path_to_files.iterdir())[0]
    std_class, std_glob = _return_std(image_path=path_file, mean=_return_mean(path_file))
    assert std_glob == 316063.0
    assert np.allclose(std_class, [38926.12389586361, 36001.38344827261, 30250.40256187894], rtol=2e-02)
