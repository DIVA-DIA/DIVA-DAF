from src.models.backbones.VGG import vgg11, vgg19_bn, vgg19, vgg16_bn, vgg16, vgg13_bn, vgg13, vgg11_bn
import torch


def test_vgg11():
    model = vgg11(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_vgg11_bn():
    model = vgg11_bn(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_vgg13():
    model = vgg13(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_vgg13_bn():
    model = vgg13_bn(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_vgg16():
    model = vgg16(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_vgg16_bn():
    model = vgg16_bn(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_vgg19():
    model = vgg19(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()


def test_vgg19_bn():
    model = vgg19_bn(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()
