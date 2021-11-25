from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image


def save_output_page_image(image_name, output_image, output_folder: Path, class_encoding: List[Tuple[int]]):
    """
    Helper function to save the output during testing in the DIVAHisDB format

    Parameters
    ----------
    image_name: str
        name of the image that is saved
    output_image: numpy matrix of size [#C x H x W]
        output image at full size
    output_folder: Path
        path to the output folder for the test data
    class_encoding: list(tuple(int))
        list with the class encodings

    Returns
    -------
    mean_iu : float
        mean iu of this image
    """

    output_encoded = output_to_class_encodings(output_image, class_encoding)

    dest_folder = output_folder
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_filename = dest_folder / image_name

    # Save the output
    Image.fromarray(output_encoded.astype(np.uint8)).save(str(dest_filename))


def output_to_class_encodings(output, class_encodings):
    """
    This function converts the output prediction matrix to an image like it was provided in the ground truth

    Parameters
    -------
    output : np.array of size [#C x H x W]
        output prediction of the network for a full-size image, where #C is the number of classes
    class_encodings : List
        Contains the range of encoded classes
    perform_argmax : bool
        perform argmax on input data
    Returns
    -------
    numpy array of size [C x H x W] (BGR)
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
