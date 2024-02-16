import torch


def argmax_onehot(tensor: torch.Tensor):
    """
    Returns the argmax of a one-hot encoded tensor.

    :param tensor: The one-hot encoded tensor
    :type tensor: torch.Tensor
    :returns: The argmax of the one-hot encoded tensor
    :rtype: torch.Tensor
    """
    return torch.LongTensor(torch.argmax(tensor, dim=0))
