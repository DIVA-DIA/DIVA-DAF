import numpy as np

from tests.datamodules.RolfFormat.datasets.test_full_page_dataset import _get_dataspecs
from src.datamodules.RolfFormat.datasets.dataset import DatasetRolfFormat
from src.datamodules.RolfFormat.utils.image_analytics import get_analytics_data, get_analytics_gt
from src.datamodules.utils.misc import ImageDimensions, get_image_dims
from tests.test_data.dummy_data_rolf.dummy_data import data_dir

TEST_JSON_DATA = {'mean': [0.857888280095024, 0.729052895463741, 0.6279161697230975],
                  'std': [0.22418223754018474, 0.21583317527112408, 0.1944047822539466]}

TEST_JSON_GT = {
    'class_weights': [4.383550114498329e-07, 5.913660555884092e-05, 9.62084259339433e-06, 1.786735277301315e-05,
                      3.341017673983495e-05, 1.0813850379566148e-05],
    'class_encodings': [(0, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0)]}


def test_get_analytics_data(data_dir):
    img_gt_path_list = DatasetRolfFormat.get_img_gt_path_list(
        list_specs=[_get_dataspecs(data_root=data_dir, train=True)])
    analytics_data = get_analytics_data(img_gt_path_list=img_gt_path_list)

    assert np.allclose(TEST_JSON_DATA['mean'], analytics_data['mean'], rtol=2e-03)
    assert np.allclose(TEST_JSON_DATA['std'], analytics_data['std'], rtol=2e-02)


def test_get_analytics_gt(data_dir):
    img_gt_path_list = DatasetRolfFormat.get_img_gt_path_list(
        list_specs=[_get_dataspecs(data_root=data_dir, train=True)])
    analytics_gt = get_analytics_gt(img_gt_path_list=img_gt_path_list)
    assert np.array_equal(np.round(TEST_JSON_GT['class_weights'], 3), np.round(analytics_gt['class_weights'], 3))
    assert np.array_equal(TEST_JSON_GT['class_encodings'], analytics_gt['class_encodings'])


def test_get_image_dims(data_dir):
    img_gt_path_list = DatasetRolfFormat.get_img_gt_path_list(
        list_specs=[_get_dataspecs(data_root=data_dir, train=True)])
    dims = get_image_dims(data_gt_path_list=img_gt_path_list)
    assert dims == ImageDimensions(960, 1344)
