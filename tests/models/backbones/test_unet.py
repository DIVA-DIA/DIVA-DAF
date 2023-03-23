import torch

from src.models.backbones.unet import UNet, Baby_UNet, UNet16, UNet32, UNet64, OldUNet


def test_unet():
    model = UNet()
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 64, 32, 32])
    assert not output_tensor.isnan().any()


def test_old_unet():
    model = OldUNet(num_classes=3)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 3, 32, 32])
    assert not output_tensor.isnan().any()


def test_baby_unet():
    model = Baby_UNet()
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 32, 32, 32])
    assert not output_tensor.isnan().any()


def test_unet16_najoua():
    model = UNet16(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_unet32_najoua():
    model = UNet32(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()


def test_unet64_najoua():
    model = UNet64(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 5, 32, 32])
    assert not output_tensor.isnan().any()
