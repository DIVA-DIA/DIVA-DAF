from pathlib import Path
from typing import Any, Dict

from src.datamodules.utils.misc import check_missing_analytics, save_json
from src.datamodules.utils.image_analytics import compute_mean_std
from src.utils import utils

log = utils.get_logger(__name__)


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
        log.info("Analytics data found and loaded!")
        return analytics_data

    log.info("No analytics data found!")

    train_path = input_path / 'train'
    image_extension = next(train_path.glob("*/*")).suffix
    gt_data_path_list = list(train_path.glob(f'**/*{image_extension}'))

    mean, std = compute_mean_std(file_names=gt_data_path_list)
    analytics_data = {'mean': mean.tolist(),
                      'std': std.tolist()}
    # save json
    save_json(analytics_data, analytics_path_data)

    return analytics_data
