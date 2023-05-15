#!/usr/bin/env bash

set -e

weights_50=("/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-30-16/checkpoints/epoch\=21/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-44-37/checkpoints/epoch\=34/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/19-58-31/checkpoints/epoch\=29/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_luca/hip/rlsa_csg18_sauvola_unet_loss_no_weights_50epoch/2023-03-31/20-12-24/checkpoints/epoch\=26/backbone.pth"
)

for j in "${!weights_50[@]}"; do
  params="experiment=fine_tune_csg18_polygon_unet.yaml
          trainer.devices=[0,1,2,3]
          mode=hip.yaml
          +model.backbone.path_to_weights=${weights_50[$j]}
          datamodule.data_dir=/net/research-hisdoc/datasets/semantic_segmentation/datasets/polygon_gt/CSG18/1152_1728
          name=fine_tune_sauvola_rlsa_csg18_polygon_unet_loss_no_weights_50pt_100e_1152_1728
          logger.wandb.tags=[unet,csg18,polygon,3-classes,fine-tune,100-epochs,no-weights,sauvola_rlsa,50-epoch-pt,with_header]
          logger.wandb.project=hip
          logger.wandb.group=fine-tune-sauvola_rlsa-3cl-polygon-csg18-50pt"
  python run.py ${params}
  #    echo ${params}
done