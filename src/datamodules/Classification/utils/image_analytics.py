import errno
import json
import logging
from pathlib import Path
from typing import Any, Dict

from datamodules.utils.misc import check_missing_analytics, save_json
from src.datamodules.utils.image_analytics import compute_mean_std


def get_analytics_data_image_folder(input_path: Path) -> Dict[str, Any]:
    """
    Computes mean and std of the images in the input_path folder.

    :param input_path: Path to the root of the dataset
    :type input_path: Path
    :return: Dictionary with mean and std
    :rtype: Dict[str, Any]
    """
    expected_keys_data = ['mean', 'std']

    analytics_path_data = input_path / 'analytics.data.train.json'

    analytics_data, missing_analytics_data = check_missing_analytics(analytics_path_data, expected_keys_data)

    if not missing_analytics_data:
        return analytics_data

    train_path = input_path / 'train'
    gt_data_path_list = list(train_path.glob('**/*.png'))

    mean, std = compute_mean_std(file_names=gt_data_path_list)
    analytics_data = {'mean': mean.tolist(),
                      'std': std.tolist()}
    # save json
    save_json(analytics_data, analytics_path_data)

    return analytics_data
