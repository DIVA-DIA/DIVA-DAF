from typing import List

import torch
from torch.nn.functional import one_hot


def gt_to_int_encoding(matrix: torch.Tensor, class_encodings: torch.Tensor):
    """
    Convert ground truth tensor or numpy matrix to one-hot encoded matrix

    :param matrix: Image as a tensor of size [C x H x W] (BGR)
    :type matrix: torch.Tensor
    :param class_encodings: class encoding so which class (index) has what value (element)
    :type class_encodings: List[int]
    :return: integer encoded matrix of size [#C x H x W]
    :rtype: torch.Tensor
    """
    integer_encoded = torch.full(size=matrix[0].shape, fill_value=-1, dtype=torch.long)
    for index, encoding in enumerate(class_encodings):
        mask = torch.logical_and(torch.logical_and(
            torch.where(matrix[0] == encoding[0], True, False),
            torch.where(matrix[1] == encoding[1], True, False)),
            torch.where(matrix[2] == encoding[2], True, False))
        integer_encoded[mask] = index

    return integer_encoded


def gt_to_one_hot(matrix: torch.Tensor, class_encodings: torch.Tensor):
    """
    Convert ground truth tensor or numpy matrix to one-hot encoded matrix

    :param matrix: float tensor from to_tensor() or numpy array
        shape (C x H x W) in the range [0.0, 1.0] or shape (H x W x C) BGR
    :type matrix: torch.Tensor or np.ndarray
    :param class_encodings: List of int
        Blue channel values that encode the different classes
    :type class_encodings: List[int]
    :return: Tensor of size [#C x H x W]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    :rtype: torch.LongTensor
    """
    integer_encoded = gt_to_int_encoding(matrix=matrix, class_encodings=class_encodings)

    num_classes = class_encodings.shape[0]

    onehot_encoded = one_hot(input=integer_encoded, num_classes=num_classes)
    onehot_encoded = onehot_encoded.swapaxes(1, 2).swapaxes(0, 1)  # changes axis from (0, 1, 2) to (2, 0, 1)

    return onehot_encoded
