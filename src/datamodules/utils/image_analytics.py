import logging
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def compute_mean_std(file_names: np.ndarray, inmem=False, workers=4):
    """
    Computes mean and std of all images present at target folder.

    Parameters
    ----------
    input_folder : String (path)
        Path to the dataset folder (see above for details)
    inmem : Boolean
        Specifies whether is should be computed i nan online of offline fashion.
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
    mean : float
        Mean value of all pixels of the images in the input folder
    std : float
        Standard deviation of all pixels of the images in the input folder
    """
    file_names_np = np.array(list(map(str, file_names)))
    # Compute mean and std
    mean, std = _cms_inmem(file_names_np) if inmem else _cms_online(file_names_np, workers)
    return mean, std


def _cms_online(file_names, workers=4):
    """
    Computes mean and image_classification deviation in an online fashion.
    This is useful when the dataset is too big to be allocated in memory.

    Parameters
    ----------
    file_names : List of String
        List of file names of the dataset
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
    mean : double
    std : double
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


def _return_mean(image_path):
    img = np.array(Image.open(image_path).convert('RGB'))
    mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])]) / 255.0
    return mean


def _return_std(image_path, mean):
    img = np.array(Image.open(image_path).convert('RGB')) / 255.0
    m2 = np.square(np.array([img[:, :, 0] - mean[0], img[:, :, 1] - mean[1], img[:, :, 2] - mean[2]]))
    return np.sum(np.sum(m2, axis=1), 1), m2.size / 3.0


def _cms_inmem(file_names):
    """
    Computes mean and image_classification deviation in an offline fashion. This is possible only when the dataset can
    be allocated in memory.

    Parameters
    ----------
    file_names: List of String
        List of file names of the dataset
    Returns
    -------
    mean : double
    std : double
    """
    img = np.zeros([file_names.size] + list(np.array(Image.open(file_names[0]).convert('RGB')).shape))

    # Load all samples
    for i, sample in enumerate(file_names):
        img[i] = np.array(Image.open(sample).convert('RGB'))

    mean = np.array([np.mean(img[:, :, :, 0]), np.mean(img[:, :, :, 1]), np.mean(img[:, :, :, 2])]) / 255.0
    std = np.array([np.std(img[:, :, :, 0]), np.std(img[:, :, :, 1]), np.std(img[:, :, :, 2])]) / 255.0

    return mean, std