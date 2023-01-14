from pathlib import Path
from typing import List, Generator

import numpy as np
from PIL import ImageOps, Image
from skimage.filters.thresholding import threshold_otsu, threshold_niblack, threshold_sauvola, gaussian
from tqdm import tqdm


def difference_of_gaussians(img: Image, sigma1: int, sigma2: int):
    """
    Difference of Gaussians
    :param img: Image object
    :param sigma1: Sigma for first Gaussian
    :param sigma2: Sigma for second Gaussian
    :return: Image object (Gaussian2 - Gaussian1)
    """
    blur1 = gaussian(img, sigma1)
    blur2 = gaussian(img, sigma2)
    return blur2 - blur1


def get_binary_threshold(gray_img_array: np.ndarray, bin_algo: str):
    if bin_algo == 'otsu':
        threshold = threshold_otsu(gray_img_array)
    elif bin_algo == 'niblack':
        threshold = threshold_niblack(gray_img_array, window_size=2005, k=0.2)
    elif bin_algo == 'sauvola':
        threshold = threshold_sauvola(gray_img_array, window_size=15, k=0.2, r=None)
    else:
        raise ValueError('Unknown binarization algorithm')
    return threshold


def binarize_images(img_paths: List[Path], global_output_dir: Path, bin_algo: str, boarder_filter: int):
    output_dir_path = global_output_dir / bin_algo / "all_files" / "tmp"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(img_paths):
        image = Image.open(img_path)
        gray_scale = ImageOps.grayscale(image)
        img_array = np.asarray(gray_scale)
        if bin_algo != 'gaussian':
            threshold = get_binary_threshold(gray_img_array=img_array, bin_algo=bin_algo)
            mask = img_array > threshold
        else:
            gaussian = difference_of_gaussians(img_array, 1, 40)
            mask = (1 - gaussian) > get_binary_threshold(1 - gaussian, bin_algo='otsu')
        if boarder_filter:
            mask[img_array < boarder_filter] = 255
        Image.fromarray(mask).save(output_dir_path / img_path.name)


if __name__ == '__main__':
    codex_path = Path("/net/research-hisdoc/datasets/self-supervised/CB55/resized/960_1344/filtered")
    output_path = Path("/net/research-hisdoc/datasets/self-supervised/CB55/binary_cleaned")
    img_files = list(codex_path.glob("*.png"))
    boarder_filter_value = 50

    if not img_files:
        raise ValueError("Input path does not contain any png images")
    output_path.mkdir(parents=True, exist_ok=True)

    # binarize_images(img_files, output_path, bin_algo='otsu')
    # binarize_images(img_files, output_path, bin_algo='niblack')
    binarize_images(img_files, output_path, bin_algo='sauvola', boarder_filter=boarder_filter_value)
    # binarize_images(img_files, output_path, bin_algo='gaussian')
