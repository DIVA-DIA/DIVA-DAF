import torch

from src.models.headers.fully_convolution import ResNetFCNHead


def test_res_net_fcnhead():
    model = ResNetFCNHead(in_channels=512, num_classes=4, output_dims=(12, 12))
    model.eval()
    output_tensor = model(torch.rand(1, 512, 1, 1))
    assert output_tensor.shape == torch.Size([1, 4, 12, 12])
    assert not output_tensor.isnan().any()
