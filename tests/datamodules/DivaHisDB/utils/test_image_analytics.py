import json

import numpy as np

from src.datamodules.DivaHisDB.datasets.cropped_dataset import CroppedHisDBDataset
from src.datamodules.DivaHisDB.utils.image_analytics import get_analytics, get_class_weights
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

TEST_JSON_DATA = {'mean': [0.7050454974582426, 0.6503181590413943, 0.5567698583877997],
                  'std': [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]}

TEST_JSON_GT = {'class_weights': [0.004952207651647859, 0.07424270397485577, 0.8964025044572563, 0.02440258391624002],
                'class_encodings': [1, 2, 4, 8]}

DATA_FOLDER_NAME = 'data'
GT_FOLDER_NAME = 'gt'
DATA_ANALYTICS_FILENAME = f'analytics.data.{DATA_FOLDER_NAME}.json'
GT_ANALYTICS_FILENAME = f'analytics.gt.hisDB.{GT_FOLDER_NAME}.json'


def test_get_analytics_no_file(data_dir_cropped):
    analytics_data, analytics_gt = get_analytics(input_path=data_dir_cropped,
                                                 data_folder_name=DATA_FOLDER_NAME, gt_folder_name=GT_FOLDER_NAME,
                                                 get_gt_data_paths_func=CroppedHisDBDataset.get_gt_data_paths)

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
