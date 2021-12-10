import torch

from src.datamodules.utils.functional import argmax_onehot


def test_argmax_onehot():
    input_tensor = torch.tensor([[[0.3143, 0.0669, 0.1640],
                                  [0.0879, 0.5411, 0.6898],
                                  [0.6721, 0.0067, 0.8442]],

                                 [[0.9010, 0.4724, 0.5524],
                                  [0.3801, 0.0128, 0.9148],
                                  [0.4442, 0.5979, 0.0799]],

                                 [[0.2001, 0.1833, 0.2530],
                                  [0.2701, 0.1324, 0.8836],
                                  [0.3645, 0.0853, 0.6570]],

                                 [[0.3610, 0.8202, 0.4263],
                                  [0.2144, 0.6923, 0.7871],
                                  [0.7420, 0.1643, 0.9310]]])
    output_tensor = argmax_onehot(input_tensor)
    assert output_tensor.shape == torch.Size([3, 3])
    assert torch.equal(output_tensor, torch.tensor([[1, 3, 1], [1, 3, 1], [3, 1, 3]]))
