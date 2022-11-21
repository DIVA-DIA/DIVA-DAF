import json

import numpy as np

from src.datamodules.RGB.datasets.cropped_dataset import CroppedDatasetRGB
from src.datamodules.RGB.utils.image_analytics import get_analytics, get_class_weights
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

TEST_JSON_DATA = {'mean': [0.7050454974582426, 0.6503181590413943, 0.5567698583877997],
                  'std': [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]}

TEST_JSON_GT = {
    'class_weights': [1.181622462170357e-06, 2.886086178533291e-05, 0.00036469730123997083, 8.845095836613389e-06,
                      0.0015267175572519084, 4.5777065690089265e-05, 0.0005162622612287042, 1.700073103143435e-05],
    'class_encodings': [(0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 0, 8), (128, 0, 1), (128, 0, 2), (128, 0, 4), (128, 0, 8)]}

TRAIN_FOLDER_NAME = 'train'
DATA_FOLDER_NAME = 'data'
GT_FOLDER_NAME = 'gt'
DATA_ANALYTICS_FILENAME = f'analytics.data.{DATA_FOLDER_NAME}.{TRAIN_FOLDER_NAME}.json'
GT_ANALYTICS_FILENAME = f'analytics.gt.{GT_FOLDER_NAME}.{TRAIN_FOLDER_NAME}.json'


def test_get_analytics_no_file(data_dir_cropped):
    analytics_data, analytics_gt = get_analytics(input_path=data_dir_cropped,
                                                 data_folder_name=DATA_FOLDER_NAME, gt_folder_name=GT_FOLDER_NAME,
                                                 train_folder_name=TRAIN_FOLDER_NAME,
                                                 get_img_gt_path_list_func=CroppedDatasetRGB.get_gt_data_paths)

    assert np.array_equal(np.round(TEST_JSON_DATA['mean'], 8), np.round(analytics_data['mean'], 8))
    assert np.array_equal(np.round(TEST_JSON_DATA['std'], 8), np.round(analytics_data['std'], 8))
    assert np.array_equal(np.round(TEST_JSON_GT['class_weights'], 8), np.round(analytics_gt['class_weights'], 8))
    assert np.array_equal(TEST_JSON_GT['class_encodings'], analytics_gt['class_encodings'])
    assert (data_dir_cropped / DATA_ANALYTICS_FILENAME).exists()
    assert (data_dir_cropped / GT_ANALYTICS_FILENAME).exists()


def test_get_analytics_load_from_file(data_dir_cropped):
    analytics_path = data_dir_cropped / DATA_ANALYTICS_FILENAME
    with analytics_path.open(mode='w') as f:
        json.dump(obj=TEST_JSON_DATA, fp=f)
    assert analytics_path.exists()

    analytics_path = data_dir_cropped / GT_ANALYTICS_FILENAME
    with analytics_path.open(mode='w') as f:
        json.dump(obj=TEST_JSON_GT, fp=f)
    assert analytics_path.exists()

    test_get_analytics_no_file(data_dir_cropped=data_dir_cropped)


def test_get_class_weights(data_dir_cropped):
    weights = get_class_weights(input_folder=data_dir_cropped)
    assert np.array_equal(weights, [0.2857142857142857, 0.35714285714285715, 0.35714285714285715])
