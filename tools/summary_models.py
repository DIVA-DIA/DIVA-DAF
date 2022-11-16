import torch
from torch.nn import Identity
from torchsummary import summary
from torchvision.models.segmentation import fcn_resnet50

from src.models.backbone_header_model import BackboneHeaderModel
from src.models.backbones import ResNet50
from src.models.backbones.divanet import DivaNet
from src.models.backbones.unet import UNet
from src.models.headers.fully_connected import ResNetHeader
from src.models.headers.fully_convolution import ResNetFCNHead

if __name__ == '__main__':
    # model = fcn_resnet50(num_classes=4, weights_backbone=None)
    model = BackboneHeaderModel(backbone=UNet(num_classes=4), header=Identity())
    model.to(torch.device('cuda'))
    summary(model, (3, 244, 244))
