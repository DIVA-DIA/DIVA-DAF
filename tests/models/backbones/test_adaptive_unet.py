import torch

from src.models.backbones.adaptive_unet import Adaptive_Unet


def test_forward():
    model = Adaptive_Unet(out_channels=3)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 3, 32, 32])
    assert not output_tensor.isnan().any()
