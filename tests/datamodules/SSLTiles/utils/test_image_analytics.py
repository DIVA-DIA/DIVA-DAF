import numpy as np

from src.datamodules.SSLTiles.utils.image_analytics import get_analytics_data_image_folder
from tests.test_data.dummy_data_histdb_new.dummy_data import data_dir_classification


def test_get_analytics_data_image_folder(data_dir_classification):
    analytics = get_analytics_data_image_folder(input_path=data_dir_classification)
    assert np.allclose(analytics['mean'], [0.898379048623119, 0.8505639432989426, 0.7596654105934847], atol=1.e-3)
    assert np.allclose(analytics['std'], [0.117466904859675, 0.1334535807597187, 0.15097226184427293], atol=1.e-3)
