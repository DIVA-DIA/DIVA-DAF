import numpy as np


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
    x2, y2 = x1 + patch.shape[2], y1 + patch.shape[1]

    # If this triggers it means that a patch is 'out-of-bounds' of the image and that should never happen!
    assert x2 <= full_output.shape[2]
    assert y2 <= full_output.shape[1]

    mask = np.isnan(full_output[:, y1:y2, x1:x2])
    # if still NaN in full_output just insert value from crop, if there is a value then take max
    full_output[:, y1:y2, x1:x2] = np.where(mask, patch, np.maximum(patch, full_output[:, y1:y2, x1:x2]))

    return full_output