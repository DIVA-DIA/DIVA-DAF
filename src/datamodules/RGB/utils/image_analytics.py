# Utils
import errno
import json
import logging
import os
from pathlib import Path

import numpy as np
# Torch related stuff
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import pil_loader

from src.datamodules.utils.image_analytics import compute_mean_std


def get_analytics(input_path: Path, data_folder_name: str, gt_folder_name: str, get_img_gt_path_list_func, **kwargs):
    """
    Parameters
    ----------
    input_path: Path to dataset

    Returns
    -------
    """
    expected_keys_data = ['mean', 'std', 'width', 'height']
    expected_keys_gt = ['class_weights', 'class_encodings']

    analytics_path_data = input_path / f'analytics.data.{data_folder_name}.json'
    analytics_path_gt = input_path / f'analytics.gt.{gt_folder_name}.json'

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
        train_path = input_path / 'train'
        img_gt_path_list = get_img_gt_path_list_func(train_path, data_folder_name=data_folder_name,
                                                     gt_folder_name=gt_folder_name)
        file_names_data = np.asarray([str(item[0]) for item in img_gt_path_list])
        file_names_gt = np.asarray([str(item[1]) for item in img_gt_path_list])

        if missing_analytics_data:
            mean, std = compute_mean_std(file_names=file_names_data, **kwargs)
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
            class_weights, class_encodings = _get_class_frequencies_weights_segmentation(gt_images=file_names_gt,
                                                                                         **kwargs)
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


def get_class_weights(input_folder, workers=4, **kwargs):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    input_folder : String (path)
        Path to the dataset folder (see above for details)
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
    ndarray[double] of size (num_classes)
        The weights vector as a 1D array normalized (sum up to 1)
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


def _get_class_frequencies_weights_segmentation(gt_images, **kwargs):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    gt_images: list of strings
        Path to all ground truth images, which contain the pixel-wise label
    workers: int
        Number of workers to use for the mean/std computation

    Returns
    -------
    ndarray[double] of size (num_classes) and ints the classes are represented as
        The weights vector as a 1D array normalized (sum up to 1)
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
    class_weights = (1 / num_samples_per_class) / ((1 / num_samples_per_class).sum())
    return class_weights.tolist(), classes


if __name__ == '__main__':
    print(get_analytics(input_path=Path('tests/dummy_data/dummy_dataset')))
