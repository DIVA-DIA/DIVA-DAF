#!/usr/bin/env bash
set -e

weightspool="/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_pool/2021-12-16/17-28-04/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_pool/2021-12-16/18-07-05/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_pool/2021-12-16/18-46-26/checkpoints/backbone_last.pth"
weightsconvpool="/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_convpool/2021-12-16/13-24-34/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_convpool/2021-12-16/14-35-46/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_convpool/2021-12-16/15-47-32/checkpoints/backbone_last.pth"

for weight in ${weightspool}
do
  params="+model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=finetune-pool-bfreeze trainer.gpus=[0,1,2,3] train=False"
  python run.py experiment=cb55_select_train30_unet_finetune_rotnet     ${params}
done

for weight in ${weightsconvpool}
do
  params="+model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=finetune-pool-bfreeze trainer.gpus=[0,1,2,3] train=False"
  python run.py experiment=cb55_select_train30_unet_finetune_rotnet     ${params}
done
