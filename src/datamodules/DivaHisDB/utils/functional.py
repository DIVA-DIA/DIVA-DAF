from typing import List

import numpy as np
import torch

from sklearn.preprocessing import OneHotEncoder


def gt_to_int_encoding(matrix: torch.Tensor, class_encodings: List[int]):
    np_array = (matrix * 255).numpy().astype(np.uint8)

    # take only blue channel
    im_np = np_array[2, :, :].astype(np.uint8)

    # change border pixels to background
    border_mask = np_array[0, :, :].astype(np.uint8) != 0
    im_np[border_mask] = 1

    im_tensor = torch.tensor(im_np)

    integer_encoded = torch.full(size=im_tensor.shape, fill_value=-1, dtype=torch.long)
    for index, encoding in enumerate(class_encodings):
        mask = torch.where(im_tensor == encoding, True, False)
        integer_encoded[mask] = index

    return integer_encoded


def gt_to_one_hot(matrix: torch.Tensor, class_encodings: List[int]):
    """
    Convert ground truth tensor or numpy matrix to one-hot encoded matrix

    Parameters
    -------
    matrix: float tensor from to_tensor() or numpy array
        shape (C x H x W) in the range [0.0, 1.0] or shape (H x W x C) BGR
    class_encodings: List of int
        Blue channel values that encode the different classes
    Returns
    -------
    torch.LongTensor of size [#C x H x W]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    """
    num_classes = len(class_encodings)

    np_array = (matrix * 255).numpy().astype(np.uint8)
    im_np = np_array[2, :, :].astype(np.uint8)
    border_mask = np_array[0, :, :].astype(np.uint8) != 0
    im_np[border_mask] = 1

    integer_encoded = np.array([i for i in range(num_classes)])
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded).astype(np.int8)

    np.place(im_np, im_np == 0,
             1)  # needed to deal with 0 fillers at the borders during testing (replace with background)
    replace_dict = {k: v for k, v in zip(class_encodings, onehot_encoded)}

    # create the one hot matrix
    one_hot_matrix = np.asanyarray(
        [[replace_dict[im_np[i, j]] for j in range(im_np.shape[1])] for i in range(im_np.shape[0])]).astype(
        np.uint8)

    return torch.LongTensor(one_hot_matrix.transpose((2, 0, 1)))


def argmax_onehot(tensor: torch.Tensor):
    """
    # TODO
    """
    return torch.LongTensor(torch.argmax(tensor, dim=0))
