# Utils
import logging

import numpy as np
# Torch related stuff

from src.datamodules.RGB.utils.image_analytics import _get_class_frequencies_weights_segmentation
from src.datamodules.utils.image_analytics import compute_mean_std


def get_analytics_data(img_gt_path_list, **kwargs):
    file_names_data = np.asarray([str(item[0]) for item in img_gt_path_list])

    mean, std = compute_mean_std(file_names=file_names_data, **kwargs)
    analytics_data = {'mean': mean.tolist(),
                      'std': std.tolist()}

    return analytics_data


def get_analytics_gt(img_gt_path_list, **kwargs):
    file_names_gt = np.asarray([str(item[1]) for item in img_gt_path_list])

    # Measure weights for class balancing
    logging.info(f'Measuring class weights')
    # create a list with all gt file paths
    class_weights, class_encodings = _get_class_frequencies_weights_segmentation(gt_images=file_names_gt,
                                                                                 **kwargs)
    analytics_gt = {'class_weights': class_weights,
                    'class_encodings': class_encodings}

    return analytics_gt


