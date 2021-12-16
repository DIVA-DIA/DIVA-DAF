import torch

from src.models.backbones import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def test_res_net18():
    model = ResNet18()
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 512, 1, 1])
    assert not output_tensor.isnan().any()


def test_res_net34():
    model = ResNet34()
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 512, 1, 1])
    assert not output_tensor.isnan().any()


def test_res_net50():
    model = ResNet50()
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 2048, 1, 1])
    assert not output_tensor.isnan().any()


def test_res_net101():
    model = ResNet101()
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 2048, 1, 1])
    assert not output_tensor.isnan().any()


def test_res_net152():
    model = ResNet152()
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 2048, 1, 1])
    assert not output_tensor.isnan().any()
