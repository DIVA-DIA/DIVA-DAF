import torch

from src.models.backbones.doc_ufcn import Doc_ufcn


def test_forward():
    model = Doc_ufcn(out_channels=3)
    model.eval()
    output_tensor = model(torch.rand(1, 3, 32, 32))
    assert output_tensor.shape == torch.Size([1, 3, 32, 32])
    assert not output_tensor.isnan().any()
