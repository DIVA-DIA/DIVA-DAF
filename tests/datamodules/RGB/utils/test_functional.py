import pytest
import torch

from src.datamodules.RGB.utils.functional import gt_to_int_encoding, gt_to_one_hot


@pytest.fixture()
def input_matrix():
    return torch.as_tensor([[[255, 255, 255], [255, 255, 0], [255, 0, 255]],
                            [[255, 255, 0], [255, 0, 0], [0, 0, 255]],
                            [[255, 0, 0], [0, 0, 0], [0, 0, 255]]])


@pytest.fixture()
def class_encodings():
    return torch.as_tensor([[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255]])


def test_gt_to_int_encoding(input_matrix, class_encodings):
    output = gt_to_int_encoding(matrix=input_matrix, class_encodings=class_encodings)
    assert torch.equal(output, torch.tensor([[3, 2, 1], [2, 1, 0], [1, 0, 3]]))


def test_gt_to_one_hot(input_matrix, class_encodings):
    output = gt_to_one_hot(matrix=input_matrix, class_encodings=class_encodings)
    assert torch.equal(output, torch.tensor([[[0, 0, 0],
                                              [0, 0, 1],
                                              [0, 1, 0]],
                                             [[0, 0, 1],
                                              [0, 1, 0],
                                              [1, 0, 0]],
                                             [[0, 1, 0],
                                              [1, 0, 0],
                                              [0, 0, 0]],
                                             [[1, 0, 0],
                                              [0, 0, 0],
                                              [0, 0, 1]]]))
