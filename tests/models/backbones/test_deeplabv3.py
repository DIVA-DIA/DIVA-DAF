import torch

from src.models.backbones.deeplabv3 import deeplabv3


def test_deeplabv3():
    model = deeplabv3(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()
