# Utils
import errno
import json
import logging
from pathlib import Path

import numpy as np

from src.datamodules.utils.image_analytics import compute_mean_std


def get_analytics_data(input_path: Path, data_folder_name: str, get_gt_data_paths_func, **kwargs):
    """
    Parameters
    ----------
    input_path: Path to dataset

    Returns
    -------
    """
    expected_keys_data = ['mean', 'std']

    analytics_path_data = input_path / f'analytics.data.{data_folder_name}.json'

    analytics_data = None

    missing_analytics_data = True

    if analytics_path_data.exists():
        with analytics_path_data.open(mode='r') as f:
            analytics_data = json.load(fp=f)
        # check if analytics file is complete
        if all(k in analytics_data for k in expected_keys_data):
            missing_analytics_data = False

    if missing_analytics_data:
        train_path = input_path / 'train'
        gt_data_path_list = get_gt_data_paths_func(train_path, data_folder_name=data_folder_name, gt_folder_name=None)

        mean, std = compute_mean_std(file_names=gt_data_path_list, **kwargs)
        analytics_data = {'mean': mean.tolist(),
                          'std': std.tolist()}
        # save json
        try:
            with analytics_path_data.open(mode='w') as f:
                json.dump(obj=analytics_data, fp=f)
        except IOError as e:
            if e.errno == errno.EACCES:
                logging.warning(f'WARNING: No permissions to write analytics file ({analytics_path_data})')
            else:
                raise

    return analytics_data
