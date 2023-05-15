import torch

from src.models.backbones.deeplabv3 import deeplabv3, deeplabv3_resnet18_os16, deeplabv3_resnet34_os16, \
    deeplabv3_resnet50_os16, deeplabv3_resnet101_os16, deeplabv3_resnet152_os16, deeplabv3_resnet18_os8, \
    deeplabv3_resnet34_os8


def test_deeplabv3():
    model = deeplabv3(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_deeplabv3_resnet18_os16():
    model = deeplabv3_resnet18_os16(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_deeplabv3_resnet34_os16():
    model = deeplabv3_resnet34_os16(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_deeplabv3_resnet50_os16():
    model = deeplabv3_resnet50_os16(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_deeplabv3_resnet101_os16():
    model = deeplabv3_resnet101_os16(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_deeplabv3_resnet152_os16():
    model = deeplabv3_resnet152_os16(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_deeplabv3_resnet18_os8():
    model = deeplabv3_resnet18_os8(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_deeplabv3_resnet34_os8():
    model = deeplabv3_resnet34_os8(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()
