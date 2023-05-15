from src.models.backbones.segnet import SegNet
import torch


def test_forward():
    model = SegNet(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()
