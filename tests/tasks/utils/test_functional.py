import pytest
import torch
from _pytest.fixtures import fixture

from src.datamodules.DivaHisDB.utils.functional import gt_to_one_hot


@fixture
def get_class_encodings():
    return [1, 2]


@fixture
def get_input_tensor():
    return torch.tensor(
        [[[0.01, 0.1], [0.001, 0.01], [0.01, 0.1]], [[0.01, 0.1], [0.01, 0.1], [3.01, 0.1]],
         [[0.01, 0.1], [0.01, 0.1], [3.01, 0.1]]],
        dtype=torch.float)


def test_gt_to_one_hot_work(get_input_tensor, get_class_encodings):
    result = gt_to_one_hot(get_input_tensor, get_class_encodings)
    assert torch.equal(result, torch.tensor([[[1, 1],
                                              [0, 1],
                                              [1, 1]],
                                             [[0, 0],
                                              [1, 0],
                                              [0, 0]]]))


def test_gt_to_one_hot_crash(get_input_tensor):
    class_encodings = [1]
    with pytest.raises(KeyError):
        gt_to_one_hot(get_input_tensor, class_encodings)

