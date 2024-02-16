from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

from src.datamodules.RGB.utils.output_tools import output_to_class_encodings


def save_output_page_image(image_name, output_image, output_folder: Path, class_encoding: List[Tuple[int]]):
    """
    Helper function to save the output during testing in the DIVAHisDB format

    :param image_name: name of the image that is saved
    :type image_name: str
    :param output_image: output image at full size
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

    # create output image and put palette
    img = Image.fromarray(output_encoded.astype(np.uint8))
    palette = np.zeros(768)
    class_encoding_np = np.asarray(class_encoding)
    palette[0:len(class_encoding_np.flatten())] = class_encoding_np.flatten()
    img.putpalette(palette.tolist())
    # Save the output
    img.save(dest_filename)
