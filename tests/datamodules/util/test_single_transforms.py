import torch

from src.datamodules.utils.single_transforms import OneHotToPixelLabelling, RightAngleRotation


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


def test_right_angle_rotation():
    torch.manual_seed(1)
    transformation = RightAngleRotation()
    tensor_input = torch.Tensor([[[0.5, 0.5],
                                  [0.1, 0.1]],
                                 [[0.2, 0.2],
                                  [0.9, 0.9]]])
    trans_output = transformation(tensor=tensor_input)
    assert torch.eq(transformation.target_class, torch.Tensor([1]))
    assert torch.eq(trans_output, torch.Tensor([[[0.2, 0.5],
                                                 [0.9, 0.1]],
                                                [0.2, 0.5],
                                                [0.9, 0.1]]))


def test__update_target_class():
    torch.manual_seed(1)  # 1, 2, 0
    transformation = RightAngleRotation()
    assert torch.eq(transformation.target_class, torch.Tensor([1]))
    transformation._update_target_class()
    assert torch.eq(transformation.target_class, torch.Tensor([2]))
