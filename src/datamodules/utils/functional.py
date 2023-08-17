import torch


def argmax_onehot(tensor: torch.Tensor):
    """
    Returns the argmax of a one-hot encoded tensor.

    :returns: torch.LongTensor
    """
    return torch.LongTensor(torch.argmax(tensor, dim=0))
