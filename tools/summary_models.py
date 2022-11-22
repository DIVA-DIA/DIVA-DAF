import torch
from torch.nn import Identity
from torchinfo import summary
from torchvision.models.segmentation import fcn_resnet50

from src.models.backbone_header_model import BackboneHeaderModel
from src.models.backbones import ResNet50
from src.models.backbones.unet import UNet
from src.models.headers.fully_connected import ResNetHeader
from src.models.headers.fully_convolution import ResNetFCNHead

if __name__ == '__main__':
    # model = fcn_resnet50(num_classes=4, weights_backbone=None)
    model = UNet(num_classes=4)
    model.to(torch.device('cuda'))
    model.eval()
    model.half()  # to FP16
    summary(model, (1, 3, 1152, 1728), device='cuda', dtypes=[torch.float16])
