from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

from src.datamodules.RGB.utils.output_tools import output_to_class_encodings


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
