# Utils
import errno
import json
import logging
from pathlib import Path

import numpy as np
# Torch related stuff
from PIL import Image

from src.datamodules.utils.image_analytics import compute_mean_std
from src.datamodules.utils.misc import pil_loader_gif


def get_analytics(input_path: Path, data_folder_name: str, gt_folder_name: str, train_folder_name: str,
                  get_img_gt_path_list_func):
    """
    Parameters
    ----------
    input_path: Path to dataset

    Returns
    -------
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
            analytics_data = _get_and_save_data_analytics(analytics_path_data, file_names_data)

        if missing_analytics_gt:
            analytics_gt = _get_and_save_gt_analytics(analytics_path_gt, file_names_gt)

    return analytics_data, analytics_gt


def _get_and_save_gt_analytics(analytics_path_gt, file_names_gt):
    # Measure weights for class balancing
    logging.info(f'Measuring class weights')
    # create a list with all gt file paths
    class_weights, class_encodings = _get_class_frequencies_weights_segmentation_indexed(
        gt_images=file_names_gt)
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


def _get_and_save_data_analytics(analytics_path_data, file_names_data):
    mean, std = compute_mean_std(file_names=file_names_data)
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


def _get_class_frequencies_weights_segmentation_indexed(gt_images: np.ndarray):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    gt_images: ndarray of Paths
        Path to all ground truth images, which contain the pixel-wise label
    workers: int
        Number of workers to use for the mean/std computation

    Returns
    -------
    ndarray[double] of size (num_classes) and ints the classes are represented as
        The weights vector as a 1D array normalized (sum up to 1)
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

