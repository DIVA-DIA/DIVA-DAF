# Utils
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.datamodules.utils.misc import check_missing_analytics, save_json
from src.datamodules.utils.image_analytics import compute_mean_std


def get_analytics_data(input_path: Path, data_folder_name: str, get_gt_data_paths_func: callable, inmem=False,
                       workers=8) -> Dict[str, Any]:
    """
    Get analytics data from json file or compute it and save it to json file.

    :param input_path: path to the training set
    :type input_path: Path
    :param data_folder_name: name of the folder containing the data
    :type data_folder_name: str
    :param get_gt_data_paths_func: function to get the paths to the gt data
    :type get_gt_data_paths_func: callable
    :param inmem: Should the data be loaded fully into memory
    :type inmem: bool
    :param workers: Number of workers to be used for calculating the mean and std
    :type workers: int
    :return: analytics data
    :rtype: Dict[str, Any]
    """

    expected_keys_data = ['mean', 'std']

    analytics_path_data = input_path / f'analytics.data.{data_folder_name}.json'

    analytics_data, missing_analytics_data = check_missing_analytics(analytics_path_data, expected_keys_data)

    if not missing_analytics_data:
        return analytics_data
    train_path = input_path / 'train'
    gt_data_path_list = get_gt_data_paths_func(train_path, data_folder_name=data_folder_name, gt_folder_name=None)

    mean, std = compute_mean_std(file_names=gt_data_path_list, inmem=inmem, workers=workers)
    analytics_data = {'mean': mean.tolist(),
                      'std': std.tolist()}
    # save json
    save_json(analytics_data, analytics_path_data)

    return analytics_data
