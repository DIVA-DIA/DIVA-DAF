from src.models.backbones.resnetdd import resnet18, resnet34, resnet50, resnet101, resnet152
import torch


def test_ResNet18_dd():
    model = resnet18(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_ResNet34_dd():
    model = resnet34(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_ResNet50_dd():
    model = resnet50(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_ResNet101_dd():
    model = resnet101(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_ResNet152_dd():
    model = resnet152(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()
