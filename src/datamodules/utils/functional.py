import torch


def argmax_onehot(tensor: torch.Tensor):
    """
    # TODO
    """
    return torch.LongTensor(torch.argmax(tensor, dim=0))