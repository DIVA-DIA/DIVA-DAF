# Utils
import errno
import json
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Union

import numpy as np
# Torch related stuff
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import pil_loader

from src.datamodules.utils.image_analytics import compute_mean_std


def get_analytics(input_path: Path, data_folder_name: str, gt_folder_name: str, train_folder_name: str,
                  get_img_gt_path_list_func: callable, inmem: bool = False, workers: int = 8) \
        -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get the analytics for the dataset. If the analytics file is not complete, it will be computed and saved.

    :param workers: The amount of workers to use for the mean/std computation
    :type workers: int
    :param inmem: If the dataset should be loaded fully into memory
    :type inmem: bool
    :param input_path: Path to the dataset folder
    :type input_path: Path
    :param data_folder_name: Name of the folder that contains the data
    :type data_folder_name: str
    :param gt_folder_name: Name of the folder that contains the ground truth
    :type gt_folder_name: str
    :param train_folder_name: Name of the folder that contains the training data
    :type train_folder_name: str
    :param get_img_gt_path_list_func: Function that returns a list of tuples with the image and gt path
    :type get_img_gt_path_list_func: callable
    :return:
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

        if missing_analytics_gt:
            # Measure weights for class balancing
            logging.info(f'Measuring class weights')
            # create a list with all gt file paths
            class_weights, class_encodings = _get_class_frequencies_weights_segmentation(gt_images=file_names_gt)
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

    return analytics_data, analytics_gt


def get_class_weights(input_folder: Path, workers=4) -> np.ndarray:
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    :param input_folder: Path to the dataset folder (see above for details)
    :type input_folder: Path
    :param workers: Number of workers to use for the mean/std computation
    :type workers: int
    :return: The weights vector as a 1D array normalized (sum up to 1)
    :rtype: ndarray[double]

    """
    # Sanity check on the folder
    if not os.path.isdir(input_folder):
        logging.error(f"Folder {input_folder} does not exist")
        raise FileNotFoundError

    # Load the dataset
    ds = datasets.ImageFolder(input_folder, transform=transforms.Compose([transforms.ToTensor()]))

    logging.info('Begin computing class frequencies weights')

    if hasattr(ds, 'targets'):
        labels = ds.targets
    elif hasattr(ds, 'labels'):
        labels = ds.labels
    else:
        # This is a fail-safe net in case a custom dataset changed the name of the internal variables
        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=workers)
        labels = []
        for target, label in data_loader:
            labels.append(label)
        labels = np.concatenate(labels).reshape(len(ds))

    class_support = np.unique(labels, return_counts=True)[1]
    class_frequencies = class_support / len(labels)
    # Class weights are the inverse of the class frequencies
    class_weights = 1 / class_frequencies
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    class_weights /= class_weights.sum()

    logging.info('Finished computing class frequencies weights ')
    logging.info(f'Class frequencies (rounded): {np.around(class_frequencies * 100, decimals=2)}')
    logging.info(f'Class weights (rounded): {np.around(class_weights * 100, decimals=2)}')

    return class_weights


def _get_class_frequencies_weights_segmentation(gt_images: Union[np.ndarray, List[str]]) -> Tuple[np.ndarray, List[int]]:
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    :param gt_images: Path to all ground truth images, which contain the pixel-wise label
    :type gt_images: List[str]
    :return: The weights vector as a 1D array normalized (sum up to 1)
    :rtype: Tuple[np.ndarray, List[int]]
    """
    logging.info('Begin computing class frequencies weights')

    total_num_pixels = 0
    label_counter = {}

    for path in gt_images:
        img_raw = pil_loader(path)
        colors = img_raw.getcolors()

        for count, color in colors:
            total_num_pixels += count
            label_counter[color] = label_counter.get(color, 0) + count

    classes = sorted(label_counter.keys())
    num_samples_per_class = np.asarray([label_counter[k] for k in classes])
    logging.info('Finished computing class frequencies weights')
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    class_weights = (1 / num_samples_per_class)  # / ((1 / num_samples_per_class).sum())
    return class_weights.tolist(), classes


if __name__ == '__main__':
    print(get_analytics(input_path=Path('tests/dummy_data/dummy_dataset')))
