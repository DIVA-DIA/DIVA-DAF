import logging
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
from PIL import Image
from numpy import ndarray, dtype, floating


def compute_mean_std(file_names: np.ndarray[str], inmem=False, workers=8) -> Tuple[float, float]:
    """
    Computes mean and std of all images present at target folder.

    :param file_names: List of the file names of the images
    :type file_names: np.ndarray[str]
    :param inmem: Specifies whether is should be computed i nan online of offline fashion.
    :type inmem: bool
    :param workers: Number of workers to use for the mean/std computation
    :type workers: int
    :return: mean and std
    :rtype: Tuple[float, float]
    """
    file_names_np = np.array(list(map(str, file_names)))
    # Compute mean and std
    mean, std = _cms_inmem(file_names_np) if inmem else _cms_online(file_names_np, workers)
    return mean, std


def _cms_online(file_names: np.ndarray[str], workers=4) -> Tuple[float, float]:
    """
    Computes mean and image_classification deviation in an online fashion.
    This is useful when the dataset is too big to be allocated in memory.

    :param file_names: List of file names of the dataset
    :type file_names: np.ndarray[str]
    :param workers: Number of workers to use for the mean/std computation
    :type workers: int

    :returns: mean and std
    :rtype: Tuple[float, float]
    """
    logging.info('Begin computing the mean')

    # Set up a pool of workers
    pool = Pool(workers + 1)

    # Online mean
    results = pool.map(_return_mean, file_names)
    mean_sum = np.sum(np.array(results), axis=0)

    # Divide by number of samples in train set
    mean = mean_sum / file_names.size

    logging.info('Finished computing the mean')
    logging.info('Begin computing the std')

    # Online image_classification deviation
    results = pool.starmap(_return_std, [[item, mean] for item in file_names])
    std_sum = np.sum(np.array([item[0] for item in results]), axis=0)
    total_pixel_count = np.sum(np.array([item[1] for item in results]))
    std = np.sqrt(std_sum / total_pixel_count)
    logging.info('Finished computing the std')

    # Shut down the pool
    pool.close()

    return mean, std


def _return_mean(image_path: str) -> ndarray[Any, dtype[floating[Any]]]:
    """
    Computes mean of a single image

    :param image_path: Path to the image
    :type image_path: str
    :returns: mean
    :rtype: float
    """
    img = np.array(Image.open(image_path).convert('RGB'))
    mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])]) / 255.0
    return mean


def _return_std(image_path: str, mean: ndarray[Any, dtype[floating[Any]]]) -> Tuple[
    ndarray[Any, dtype[floating[Any]]], float]:
    """
    Computes image_classification deviation of a single image

    :param image_path: Path to the image
    :type image_path: str
    :param mean: Mean value of all pixels of the image
    :type mean: ndarray[Any, dtype[floating[Any]]]

    :returns: Standard deviation of all pixels of the image
    :rtype: Tuple[ndarray[Any, dtype[floating[Any]]], float]
    """
    img = np.array(Image.open(image_path).convert('RGB')) / 255.0
    m2 = np.square(np.array([img[:, :, 0] - mean[0], img[:, :, 1] - mean[1], img[:, :, 2] - mean[2]]))
    return np.sum(np.sum(m2, axis=1), 1), m2.size / 3.0


def _cms_inmem(file_names: np.ndarray[str]) -> Tuple[
    ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]:
    """
    Computes mean and image_classification deviation in an offline fashion. This is possible only when the dataset can
    be allocated in memory.


    :param file_names: List of file names of the dataset
    :type file_names: np.ndarray[str]

    :returns: Mean value of all pixels of the images in the input folder and the
    standard deviation of all pixels of the images in the input folder
    :rtype: Tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
    """
    img = np.zeros([file_names.size] + list(np.array(Image.open(file_names[0]).convert('RGB')).shape))

    # Load all samples
    for i, sample in enumerate(file_names):
        img[i] = np.array(Image.open(sample).convert('RGB'))

    mean = np.array([np.mean(img[:, :, :, 0]), np.mean(img[:, :, :, 1]), np.mean(img[:, :, :, 2])]) / 255.0
    std = np.array([np.std(img[:, :, :, 0]), np.std(img[:, :, :, 1]), np.std(img[:, :, :, 2])]) / 255.0

    return mean, std
