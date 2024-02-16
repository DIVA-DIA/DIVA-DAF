from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image


def save_output_page_image(image_name: str, output_image: np.ndarray, output_folder: Path,
                           class_encoding: List[Tuple[int]]) -> None:
    """
    Helper function to save the output during testing in the DIVAHisDB format

    :param image_name: name of the image that is saved
    :type image_name: str
    :param output_image: output image at full size [#C x H x W]
    :type output_image: np.ndarray
    :param output_folder: path to the output folder for the test data
    :type output_folder: Path
    :param class_encoding: list with the class encodings
    :type class_encoding: List[Tuple[int]]

    """

    output_encoded = output_to_class_encodings(output_image, class_encoding)

    dest_folder = output_folder
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_filename = dest_folder / image_name

    # Save the output
    Image.fromarray(output_encoded.astype(np.uint8)).save(str(dest_filename))


def output_to_class_encodings(output: np.ndarray, class_encodings: List[Tuple[int]]) -> np.ndarray:
    """
    This function converts the output prediction matrix to an image like it was provided in the ground truth

    :param output: output prediction of the network for a full-size image, where #C is the number of classes
    :type output: np.ndarray
    :param class_encodings: Contains the range of encoded classes
    :type class_encodings: List[Tuple[int]]

    :return: numpy array of size [C x H x W] (BGR) with the classes encoded as in the ground truth
    :rtype: np.ndarray
    """

    integer_encoded = np.argmax(output, axis=0)

    num_classes = len(class_encodings)

    masks = [integer_encoded == class_index for class_index in range(num_classes)]

    rgb = np.full((*integer_encoded.shape, 3), -1)
    for mask, color in zip(masks, class_encodings):
        rgb[:, :, 0] = np.where(mask, color[0], rgb[:, :, 0])
        rgb[:, :, 1] = np.where(mask, color[1], rgb[:, :, 1])
        rgb[:, :, 2] = np.where(mask, color[2], rgb[:, :, 2])

    return rgb
