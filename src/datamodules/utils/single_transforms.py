import math

import torch

import src.datamodules.utils.functional


class OneHotToPixelLabelling(object):
    """
    Transforms a one-hot encoded tensor to a pixel labelling tensor.

    :param tensor: The one-hot encoded tensor
    :type tensor: torch.Tensor
    :returns: The pixel labelling tensor
    :rtype: torch.Tensor
    """
    def __call__(self, tensor: torch.Tensor):
        return src.datamodules.utils.functional.argmax_onehot(tensor)
