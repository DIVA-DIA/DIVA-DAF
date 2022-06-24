import numpy as np
import json

from src.datamodules.RotNet.utils.image_analytics import get_analytics_data
from src.datamodules.RotNet.datasets.cropped_dataset import CroppedRotNet
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped

DATA_ANALYTICS_FILENAME = 'analytics.data.data.json'
TEST_JSON_DATA = {'mean': [0.7050454974582426, 0.6503181590413943, 0.5567698583877997],
                  'std': [0.3104060859619883, 0.3053311838884032, 0.28919611393432726]}


def test_get_analytics_data_no_file(data_dir_cropped):
    data_analytics = get_analytics_data(input_path=data_dir_cropped, data_folder_name='data',
                                        get_gt_data_paths_func=CroppedRotNet.get_gt_data_paths)
    assert np.array_equal(np.round(data_analytics['mean']),
                          np.round([0.7050454974582426, 0.6503181590413943, 0.5567698583877997]))
    assert np.array_equal(np.round(data_analytics['std']),
                          np.round([0.3104060859619883, 0.3053311838884032, 0.28919611393432726]))
    assert (data_dir_cropped / DATA_ANALYTICS_FILENAME).exists()


def test_get_analytics_load_from_file(data_dir_cropped):
    analytics_path = data_dir_cropped / DATA_ANALYTICS_FILENAME
    with analytics_path.open(mode='w') as f:
        json.dump(obj=TEST_JSON_DATA, fp=f)
    assert analytics_path.exists()

    test_get_analytics_data_no_file(data_dir_cropped=data_dir_cropped)
