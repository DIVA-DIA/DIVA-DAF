import torch

from src.models.headers.fully_connected import ResNetHeader, SingleLinear


def test_res_net_header():
    model = ResNetHeader(in_channels=512, num_classes=4)
    model.eval()
    output_tensor = model(torch.rand(1, 512, 1, 1))
    assert output_tensor.shape == torch.Size([1, 4])
    assert not output_tensor.isnan().any()


def test_single_linear():
    model = SingleLinear(in_channels=512, num_classes=4)
    model.eval()
    output_tensor = model(torch.rand(1, 512, 1, 1))
    assert output_tensor.shape == torch.Size([1, 4])
    assert not output_tensor.isnan().any()
