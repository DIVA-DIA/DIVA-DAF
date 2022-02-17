import numpy as np

from datamodules.RolfFormat.datasets.test_full_page_dataset import _get_dataspecs
from src.datamodules.RolfFormat.datasets.dataset import DatasetRolfFormat
from src.datamodules.RolfFormat.utils.image_analytics import get_analytics_data, get_analytics_gt, get_image_dims
from src.datamodules.utils.misc import ImageDimensions
from tests.test_data.dummy_data_rolf.dummy_data import data_dir

TEST_JSON_DATA = {'mean': [0.857888280095024, 0.729052895463741, 0.6279161697230975],
                  'std': [0.22418223754018474, 0.21583317527112408, 0.1944047822539466]}

TEST_JSON_GT = {
    'class_weights': [0.003338901797158088, 0.4504370052145281, 0.07328089741466476, 0.13609365634251125,
                      0.2544816330285547, 0.08236790620258311],
    'class_encodings': [(0, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0)]}


def test_get_analytics_data(data_dir):
    img_gt_path_list = DatasetRolfFormat.get_img_gt_path_list(
        list_specs=[_get_dataspecs(data_root=data_dir, train=True)])
    analytics_data = get_analytics_data(img_gt_path_list=img_gt_path_list)

    assert np.array_equal(np.round(TEST_JSON_DATA['mean'], 8), np.round(analytics_data['mean'], 8))
    assert np.array_equal(np.round(TEST_JSON_DATA['std'], 8), np.round(analytics_data['std'], 8))


def test_get_analytics_gt(data_dir):
    img_gt_path_list = DatasetRolfFormat.get_img_gt_path_list(
        list_specs=[_get_dataspecs(data_root=data_dir, train=True)])
    analytics_gt = get_analytics_gt(img_gt_path_list=img_gt_path_list)
    assert np.array_equal(np.round(TEST_JSON_GT['class_weights'], 8), np.round(analytics_gt['class_weights'], 8))
    assert np.array_equal(TEST_JSON_GT['class_encodings'], analytics_gt['class_encodings'])


def test_get_image_dims(data_dir):
    img_gt_path_list = DatasetRolfFormat.get_img_gt_path_list(
        list_specs=[_get_dataspecs(data_root=data_dir, train=True)])
    dims = get_image_dims(data_gt_path_list=img_gt_path_list)
    assert dims == ImageDimensions(960, 1344)
