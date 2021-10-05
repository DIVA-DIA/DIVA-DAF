import json

import numpy as np

from src.datamodules.hisDBDataModule.cropped_hisdb_dataset import CroppedHisDBDataset
from datamodules.util.analytics.image_analytics import get_analytics

TEST_JSON = {'mean': [0.7050454974582426, 0.6503181590413943, 0.5567698583877997],
             'std': [0.3104060859619883, 0.3053311838884032, 0.28919611393432726],
             'class_weights': [0.004952207651647859, 0.07424270397485577, 0.8964025044572563, 0.02440258391624002],
             'class_encodings': [1, 2, 4, 8]}


def test_get_analytics_no_file(data_dir_cropped):
    output = get_analytics(input_path=data_dir_cropped, get_gt_data_paths_func=CroppedHisDBDataset.get_gt_data_paths)

    assert np.array_equal(np.round(TEST_JSON['mean'], 8), np.round(output['mean'], 8))
    assert np.array_equal(np.round(TEST_JSON['std'], 8), np.round(output['std'], 8))
    assert np.array_equal(np.round(TEST_JSON['class_weights'], 8), np.round(output['class_weights'], 8))
    assert np.array_equal(TEST_JSON['class_encodings'], output['class_encodings'])
    assert (data_dir_cropped / 'analytics.json').exists()


def test_get_analytics_load_from_file(data_dir_cropped):
    analytics_path = data_dir_cropped / 'analytics.json'
    with analytics_path.open(mode='w') as f:
        json.dump(obj=TEST_JSON, fp=f)
    assert analytics_path.exists()

    test_get_analytics_no_file(data_dir_cropped=data_dir_cropped)
