import json

import numpy as np

from hisDBDataModule.image_folder_segmentation import ImageFolderSegmentationDataset
from hisDBDataModule.util.analytics.image_analytics import get_analytics
from tests.dummy_data.dummy_data import data_dir

TEST_JSON = {'mean': [0.66136009, 0.60807059, 0.51881776],
             'std': [0.35094071, 0.33749921, 0.30937039],
             'class_weights': [0.00648759, 0.08383551, 0.85457594, 0.05510097],
             'class_encodings': [1, 2, 4, 8]}


def test_get_analytics_no_file(data_dir):
    output = get_analytics(input_path=data_dir, get_gt_data_paths_func=ImageFolderSegmentationDataset.get_gt_data_paths)

    assert np.array_equal(np.round(TEST_JSON['mean'], 8), np.round(output['mean'], 8))
    assert np.array_equal(np.round(TEST_JSON['std'], 8), np.round(output['std'], 8))
    assert np.array_equal(np.round(TEST_JSON['class_weights'], 8), np.round(output['class_weights'], 8))
    assert np.array_equal(TEST_JSON['class_encodings'], output['class_encodings'])
    assert (data_dir / 'analytics.json').exists()


def test_get_analytics_load_from_file(data_dir):
    analytics_path = data_dir / 'analytics.json'
    with analytics_path.open(mode='w') as f:
        json.dump(obj=TEST_JSON, fp=f)
    assert analytics_path.exists()

    test_get_analytics_no_file(data_dir=data_dir)
