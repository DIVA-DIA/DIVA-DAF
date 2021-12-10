import pytest
import torch
from torchvision.datasets.folder import pil_loader

from src.datamodules.utils.single_transforms import OneHotToPixelLabelling
from tests.test_data.dummy_data_hisdb.dummy_data import data_dir_cropped


def test_one_hot_to_pixel_labelling():
    transformation = OneHotToPixelLabelling()
    tensor_input = torch.tensor([[[0.6999015212, 0.4833144546],
                                  [0.8329959512, 0.1569360495]],
                                 [[0.5571944118, 0.1454262733],
                                  [0.4641100168, 0.0191639662]],
                                 [[0.8016914725, 0.2552427053],
                                  [0.6105983257, 0.2813706994]]])
    trans_output = transformation(tensor_input)
    assert isinstance(trans_output, torch.Tensor)
    assert torch.equal(trans_output, torch.tensor([[2, 0], [0, 2]]))
