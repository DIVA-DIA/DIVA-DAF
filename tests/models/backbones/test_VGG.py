from src.models.backbones.VGG import vgg11
import torch


def test_vgg11():
    model = vgg11(num_classes=5)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 224, 224))
    assert output_tensor.shape == torch.Size([1, 5])
    assert not output_tensor.isnan().any()

