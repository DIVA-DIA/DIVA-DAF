from pathlib import Path

import numpy as np
from PIL import Image


def save_output_page_image(image_name, output_image, output_folder: Path, class_encoding):
    """
    Helper function to save the output during testing in the DIVAHisDB format

    :param image_name: name of the image that is saved
    :type image_name: str
    :param output_image: output image at full size
    :type output_image: np.array of size [#C x H x W]
    :param output_folder: path to the output folder for the test data
    :type output_folder: Path
    :param class_encoding: list with the class encodings
    :type class_encoding: list

    :return: mean iou of this image
    :rtype: float
    """

    output_encoded = output_to_class_encodings(output_image, class_encoding)

    dest_folder = output_folder
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_filename = dest_folder / image_name

    # Save the output
    Image.fromarray(output_encoded.astype(np.uint8)).save(str(dest_filename))


def output_to_class_encodings(output, class_encodings, perform_argmax=True):
    """
    This function converts the output prediction matrix to an image with the colors of the class encodings.

    :param output: output prediction of the network for a full-size image, where #C is the number of classes
    :type output: np.array of size [#C x H x W]
    :param class_encodings: Contains the range of encoded classes
    :type class_encodings: list
    :param perform_argmax: perform argmax on input data
    :type perform_argmax: bool
    :return: np.array of size [H x W] (BGR)
    """

    b_argmax = np.argmax(output, axis=0) if perform_argmax else output

    class_to_color = {i: j for i, j in enumerate(class_encodings)}

    masks = [b_argmax == old for old in class_to_color.keys()]

    for mask, (old, new) in zip(masks, class_to_color.items()):
        b_argmax = np.where(mask, new, b_argmax)

    rgb = np.dstack((np.zeros(shape=(b_argmax.shape[0], b_argmax.shape[1], 2), dtype=np.int8), b_argmax))

    return rgb
