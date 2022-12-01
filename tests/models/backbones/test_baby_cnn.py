import torch

from src.models.backbones.baby_cnn import CNN_basic


def test_forward():
    model = CNN_basic()
    model.eval()
    output_model = model(torch.rand(1, 3, 24, 24))  # B, C, W, H
    assert output_model.shape == torch.Size([1, 72, 1, 1])
    assert not output_model.isnan().any()  # checks if there are any nans in the output
