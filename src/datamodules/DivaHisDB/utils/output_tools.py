from pathlib import Path

import numpy as np
from PIL import Image


def save_output_page_image(image_name, output_image, output_folder: Path, class_encoding):
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
    class_encoding: list(int)
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


def output_to_class_encodings(output, class_encodings, perform_argmax=True):
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

    B = np.argmax(output, axis=0) if perform_argmax else output

    class_to_B = {i: j for i, j in enumerate(class_encodings)}

    masks = [B == old for old in class_to_B.keys()]

    for mask, (old, new) in zip(masks, class_to_B.items()):
        B = np.where(mask, new, B)

    rgb = np.dstack((np.zeros(shape=(B.shape[0], B.shape[1], 2), dtype=np.int8), B))

    return rgb
