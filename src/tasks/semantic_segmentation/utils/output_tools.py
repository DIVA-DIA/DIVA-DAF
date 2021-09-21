from pathlib import Path

import numpy
import numpy as np
import torch
from PIL import Image


def _get_argmax(output):
    """
    takes the biggest value from a pixel across all classes
    :param output: (Batch_size x num_classes x W x H)
        matrix with the given attributes
    :return: (Batch_size x W x H)
        matrix with the hisdb class number for each pixel
    """
    if isinstance(output, torch.Tensor):
        return torch.argmax(output, dim=1)
    if isinstance(output, np.ndarray):
        return np.argmax(output, axis=1)
    return output


def merge_patches(patch, coordinates, full_output):
    """
    This function merges the patch into the full output image
    Overlapping values are resolved by taking the max.

    Parameters
    ----------
    patch: numpy matrix of size [#classes x crop_size x crop_size]
        a patch from the larger image
    coordinates: tuple of ints
        top left coordinates of the patch within the larger image for all patches in a batch
    full_output: numpy matrix of size [#C x H x W]
        output image at full size
    Returns
    -------
    full_output: numpy matrix [#C x Htot x Wtot]
    """
    assert len(full_output.shape) == 3
    assert full_output.size != 0

    # Resolve patch coordinates
    x1, y1 = coordinates
    x2, y2 = x1 + patch.shape[1], y1 + patch.shape[2]

    # If this triggers it means that a patch is 'out-of-bounds' of the image and that should never happen!
    assert x2 <= full_output.shape[1]
    assert y2 <= full_output.shape[2]

    mask = np.isnan(full_output[:, x1:x2, y1:y2])
    # if still NaN in full_output just insert value from crop, if there is a value then take max
    full_output[:, x1:x2, y1:y2] = np.where(mask, patch, np.maximum(patch, full_output[:, x1:x2, y1:y2]))

    return full_output


def save_output_page_image(image_name, output_image, output_folder, class_encoding):
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

    dest_folder = output_folder / 'images'
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_filename = dest_folder / f'output_{image_name}'

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
