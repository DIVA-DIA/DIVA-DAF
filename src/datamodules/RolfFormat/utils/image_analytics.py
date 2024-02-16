# Utils
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
# Torch related stuff

from src.datamodules.RGB.utils.image_analytics import _get_class_frequencies_weights_segmentation
from src.datamodules.utils.image_analytics import compute_mean_std


def get_analytics_data(img_gt_path_list: List[Tuple[Path, Path]], inmem: bool = False, workers: int = 8)\
        -> Dict[str, List]:
    """
    Computes mean and std of the dataset

    :param img_gt_path_list: Images and their corresponding ground truth paths to be used for computing mean and std
    :type img_gt_path_list: List[Tuple[Path, Path]]
    :param inmem: Whether to load the images in memory or not
    :type inmem: bool
    :param workers: Number of workers to use for loading and calculating mean and std
    :type workers: int
    :return: Dictionary containing mean and std
    :rtype: dict
    """
    file_names_data = np.asarray([str(item[0]) for item in img_gt_path_list])

    mean, std = compute_mean_std(file_names=file_names_data, inmem=inmem, workers=workers)
    analytics_data = {'mean': mean.tolist(),
                      'std': std.tolist()}

    return analytics_data


def get_analytics_gt(img_gt_path_list: List[Tuple[Path, Path]]) -> Dict[str, Any]:
    """
    Computes class weights and encodings of the dataset based on the ground truth

    :param img_gt_path_list: Images and their corresponding ground truth paths to be used for computing class weights
    :type img_gt_path_list: List[Tuple[Path, Path]]
    :return: Dictionary containing class weights and encodings
    :rtype: Dict[str, float]
    """
    file_names_gt = np.asarray([str(item[1]) for item in img_gt_path_list])

    # Measure weights for class balancing
    logging.info(f'Measuring class weights')
    # create a list with all gt file paths
    class_weights, class_encodings = _get_class_frequencies_weights_segmentation(gt_images=file_names_gt)
    analytics_gt = {'class_weights': class_weights,
                    'class_encodings': class_encodings}

    return analytics_gt
