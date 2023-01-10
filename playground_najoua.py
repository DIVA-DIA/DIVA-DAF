import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning
import torch
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchvision.models.segmentation import fcn_resnet50

from src.callbacks.model_callbacks import SaveModelStateDictAndTaskCheckpoint
from src.datamodules.RGB.datamodule import DataModuleRGB
from src.models.backbone_header_model import BackboneHeaderModel
from src.models.backbones.unet import UNet
from src.tasks.RGB.semantic_segmentation import SemanticSegmentationRGB

if __name__ == '__main__':
    # resnet()
    unet_ft()