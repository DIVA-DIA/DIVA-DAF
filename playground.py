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
from src.models.headers.unet import UNetFCNHead
from src.tasks.RGB.semantic_segmentation import SemanticSegmentationRGB


def resnet_ft():
    warnings.filterwarnings("ignore")
    datamodule = DataModuleRGB(
        data_dir='/net/research-hisdoc/datasets/semantic_segmentation/datasets/CB55-splits/split2',
        num_workers=4,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        data_folder_name='data',
        gt_folder_name='gt')
    backbone = fcn_resnet50(num_classes=datamodule.num_classes, weights_backbone=None)
    state_dict = torch.load(
        f='/netscratch/experiments_lars_paul/lars/experiments/ssltiles_prebuild_cb55_whole_resnet50_reduced_lr/2022-08-30/19-39-44/checkpoints/epoch=49/backbone.pth',
        map_location='cpu')
    keys = backbone.load_state_dict(state_dict={"backbone." + k: state_dict[k] for k in state_dict.keys()},
                                    strict=False)
    model = BackboneHeaderModel(backbone=backbone, header=torch.nn.Identity())
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    now = datetime.now()
    root_output_path = Path(
        f"outputs/playground/{datetime.strftime(now, '%Y-%m-%d')}/{datetime.strftime(now, '%H-%M-%S')}")
    task = SemanticSegmentationRGB(model=model,
                                   optimizer=optimizer,
                                   loss_fn=loss,
                                   test_output_path=root_output_path / 'test_output',
                                   metric_train=torchmetrics.JaccardIndex(num_classes=datamodule.num_classes),
                                   metric_val=torchmetrics.JaccardIndex(num_classes=datamodule.num_classes),
                                   metric_test=torchmetrics.JaccardIndex(num_classes=datamodule.num_classes))
    callbacks = [SaveModelStateDictAndTaskCheckpoint(monitor="val/crossentropyloss",
                                                     save_top_k=1,
                                                     save_last=True,
                                                     mode="min",
                                                     verbose=False,
                                                     dirpath=f"{str(root_output_path)}/checkpoints/",
                                                     filename='{epoch}/task-checkpoint',
                                                     backbone_filename='{epoch}/backbone',
                                                     header_filename='{epoch}/header')]
    logger = WandbLogger(project="unsupervised",
                         name='cb55_full_split2_FCN_resnet50_finetune_complete',
                         offline=False,  # set True to store all logs only locally
                         job_type="train",  # specifies the type of run
                         save_dir=str(root_output_path),
                         log_model=False)
    trainer = Trainer(strategy='ddp', accelerator='gpu', devices=-1, logger=logger, max_epochs=100, callbacks=callbacks,
                      precision=16)
    # accumulate_grad_batches=5)
    trainer.fit(task, datamodule)
    trainer.test(task, datamodule)


def unet_ft():
    warnings.filterwarnings("ignore")
    datamodule = DataModuleRGB(
        data_dir='/net/research-hisdoc/datasets/semantic_segmentation/datasets/CB55-splits/AB1',
        num_workers=4,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        data_folder_name='data',
        gt_folder_name='gt',
        train_folder_name='training-40',
        val_folder_name='validation')
    backbone = UNet(num_classes=datamodule.num_classes)
    state_dict = torch.load(
        f='/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/morpho_cb55_B22_unet_loss_no_weights/2022-10-20/22-39-36/checkpoints/epoch=74/backbone.pth',
        map_location='cpu')
    keys = backbone.load_state_dict(state_dict={k: state_dict[k] for k in state_dict.keys() if 'layers.9' not in k},
                                    strict=False)
    model = BackboneHeaderModel(backbone=backbone, header=torch.nn.Identity())
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    now = datetime.now()
    root_output_path = Path(
        f"outputs/playground/{datetime.strftime(now, '%Y-%m-%d')}/{datetime.strftime(now, '%H-%M-%S')}")
    task = SemanticSegmentationRGB(model=model,
                                   optimizer=optimizer,
                                   loss_fn=loss,
                                   test_output_path=root_output_path / 'test_output',
                                   metric_train=torchmetrics.JaccardIndex(num_classes=datamodule.num_classes),
                                   metric_val=torchmetrics.JaccardIndex(num_classes=datamodule.num_classes),
                                   metric_test=torchmetrics.JaccardIndex(num_classes=datamodule.num_classes))
    callbacks = [SaveModelStateDictAndTaskCheckpoint(monitor="val/jaccard_index",
                                                     save_top_k=1,
                                                     save_last=True,
                                                     mode="max",
                                                     verbose=False,
                                                     dirpath=f"{str(root_output_path)}/checkpoints/",
                                                     filename='{epoch}/task-checkpoint',
                                                     backbone_filename='{epoch}/backbone',
                                                     header_filename='{epoch}/header')]
    logger = WandbLogger(project="ijdar",
                         name='morpho_fine_tune_AB1_train_40_cb55_B22_unet_loss_no_weights',
                         tags=["Morpho", "pre-training", "unet", "B22", "CB55-ssl-set", "fine-tune", "AB1", "train-40"],
                         offline=False,  # set True to store all logs only locally
                         job_type="train",  # specifies the type of run
                         save_dir=str(root_output_path),
                         log_model=False)
    trainer = Trainer(strategy='ddp', accelerator='gpu', devices=-1, logger=logger, max_epochs=100, callbacks=callbacks,
                      precision=16, accumulate_grad_batches=5)
    # accumulate_grad_batches=5)
    trainer.fit(task, datamodule)
    trainer.test(task, datamodule)


if __name__ == '__main__':
    # resnet()
    unet_ft()
