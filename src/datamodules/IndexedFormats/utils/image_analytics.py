# Utils
import errno
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
# Torch related stuff
from PIL import Image

from src.datamodules.utils.image_analytics import compute_mean_std
from src.datamodules.utils.misc import pil_loader_gif


def get_analytics(input_path: Path, data_folder_name: str, gt_folder_name: str, train_folder_name: str,
                  get_img_gt_path_list_func: callable, inmem: bool = False, workers: int = 8) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get the analytics for the dataset. If the analytics file is not present, it will be computed and saved.

    :param workers:  Number of workers to calculate the mean and std
    :type workers: int
    :param inmem:  Load the images in memory or load them separately
    :type inmem: bool
    :param input_path: Path to the root of the dataset
    :type input_path: Path
    :param data_folder_name: Name of the folder containing the data
    :type data_folder_name: str
    :param gt_folder_name: Name of the folder containing the ground truth
    :type gt_folder_name: str
    :param train_folder_name: Name of the folder containing the training data
    :type train_folder_name: str
    :param get_img_gt_path_list_func: Function to get the list of image and ground truth paths
    :type get_img_gt_path_list_func: Callable[[Path, str, str], List[Tuple[Path, Path]]]
    :return: Tuple of analytics for the data and ground truth
    :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
    """
    expected_keys_data = ['mean', 'std', 'width', 'height']
    expected_keys_gt = ['class_weights', 'class_encodings']

    analytics_path_data = input_path / f'analytics.data.{data_folder_name}.{train_folder_name}.json'
    analytics_path_gt = input_path / f'analytics.gt.{gt_folder_name}.{train_folder_name}.json'

    analytics_data = None
    analytics_gt = None

    missing_analytics_data = True
    missing_analytics_gt = True

    if analytics_path_data.exists():
        with analytics_path_data.open(mode='r') as f:
            analytics_data = json.load(fp=f)
        # check if analytics file is complete
        if all(k in analytics_data for k in expected_keys_data):
            missing_analytics_data = False

    if analytics_path_gt.exists():
        with analytics_path_gt.open(mode='r') as f:
            analytics_gt = json.load(fp=f)
        # check if analytics file is complete
        if all(k in analytics_gt for k in expected_keys_gt):
            missing_analytics_gt = False

    if missing_analytics_data or missing_analytics_gt:
        train_path = input_path / train_folder_name
        img_gt_path_list = get_img_gt_path_list_func(train_path, data_folder_name=data_folder_name,
                                                     gt_folder_name=gt_folder_name)
        file_names_data = np.asarray([str(item[0]) for item in img_gt_path_list])
        file_names_gt = np.asarray([str(item[1]) for item in img_gt_path_list])

        if missing_analytics_data:
            analytics_data = _get_and_save_data_analytics(analytics_path_data, file_names_data, inmem=inmem, workers=workers)

        if missing_analytics_gt:
            analytics_gt = _get_and_save_gt_analytics(analytics_path_gt, file_names_gt)

    return analytics_data, analytics_gt


def _get_and_save_gt_analytics(analytics_path_gt: Path, file_names_gt: np.ndarray) -> Dict[str, Any]:
    """
    Get the analytics for the ground truth. If the analytics file is not present, it will be computed and saved.

    :param analytics_path_gt: Path to the analytics file
    :type analytics_path_gt: Path
    :param file_names_gt: names of the files in the training set
    :type file_names_gt: np.ndarray
    :return: The analytics for the ground truth
    :rtype: Dict[str, Any]
    """
    # Measure weights for class balancing
    logging.info(f'Measuring class weights')
    # create a list with all gt file paths
    class_weights, class_encodings = _get_class_frequencies_weights_segmentation_indexed(gt_images=file_names_gt)
    analytics_gt = {'class_weights': class_weights,
                    'class_encodings': class_encodings}
    # save json
    try:
        with analytics_path_gt.open(mode='w') as f:
            json.dump(obj=analytics_gt, fp=f)
    except IOError as e:
        if e.errno == errno.EACCES:
            print(f'WARNING: No permissions to write analytics file ({analytics_path_gt})')
        else:
            raise
    return analytics_gt


def _get_and_save_data_analytics(analytics_path_data: Path, file_names_data: np.ndarray, inmem: bool, workers: int) -> Dict[str, Any]:
    """
    Get the analytics for the data. If the analytics file is not present, it will be computed and saved.

    :param analytics_path_data: Path to the analytics file
    :param file_names_data: names of the files in the training set
    :return: The analytics for the data
    :rtype: Dict[str, Any]
    """
    mean, std = compute_mean_std(file_names=file_names_data, inmem=inmem, workers=workers)
    img = Image.open(file_names_data[0]).convert('RGB')
    analytics_data = {'mean': mean.tolist(),
                      'std': std.tolist(),
                      'width': img.width,
                      'height': img.height}
    # save json
    try:
        with analytics_path_data.open(mode='w') as f:
            json.dump(obj=analytics_data, fp=f)
    except IOError as e:
        if e.errno == errno.EACCES:
            print(f'WARNING: No permissions to write analytics file ({analytics_path_data})')
        else:
            raise
    return analytics_data


def _get_class_frequencies_weights_segmentation_indexed(gt_images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    :param gt_images: Path to all ground truth images, which contain the pixel-wise label
    :type gt_images: np.ndarray
    :return: The weights vector as a 1D array normalized (sum up to 1)
    :rtype: np.ndarray
    """
    logging.info('Begin computing class frequencies weights')

    label_counter = {}
    color_table = []
    unique_colors = 0

    for path in gt_images:
        img_raw = pil_loader_gif(path)
        colors = img_raw.getcolors()

        if len(colors) > unique_colors:
            color_table = img_raw.getpalette()
            unique_colors = len(colors)

        for count, index in colors:
            label_counter[index] = label_counter.get(index, 0) + count

    classes = np.asarray(color_table).reshape(256, 3)[:unique_colors]
    num_samples_per_class = np.asarray(list(label_counter.values()))
    logging.info('Finished computing class frequencies weights')
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    class_weights = (1 / num_samples_per_class)  # / ((1 / num_samples_per_class).sum())
    return class_weights.tolist(), classes.tolist()
