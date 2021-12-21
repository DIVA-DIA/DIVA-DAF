#!/usr/bin/env bash
set -e

weights1="/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_train1_val1_unet_finetune_rotnet/2021-12-20/20-31-38/checkpoints/epoch=47/backbone.pth
          /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_train1_val1_unet_finetune_rotnet/2021-12-20/23-29-05/checkpoints/epoch=47/backbone.pth
          /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_train1_val1_unet_finetune_rotnet/2021-12-21/02-26-53/checkpoints/epoch=43/backbone.pth"

weights15="/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_train15_unet_finetune_rotnet/2021-12-20/20-41-10/checkpoints/epoch=39/backbone.pth
           /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_train15_unet_finetune_rotnet/2021-12-20/23-38-36/checkpoints/epoch=39/backbone.pth
           /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_train15_unet_finetune_rotnet/2021-12-21/02-36-31/checkpoints/epoch=39/backbone.pth"

weights30="/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_full_unet_finetune_rotnet/2021-12-20/21-54-47/checkpoints/epoch=47/backbone.pth
           /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_full_unet_finetune_rotnet/2021-12-21/00-52-24/checkpoints/epoch=27/backbone.pth
           /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/semantic_segmentation_cb55_full_unet_finetune_rotnet/2021-12-21/03-50-21/checkpoints/epoch=43/backbone.pth"


for weight in ${weights1}
do
  params="+model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=finetune-convpool trainer.gpus=[0,1,2,3] train=False"
  python run.py experiment=cb55_select_train1_val1_unet_finetune_rotnet ${params}
done

for weight in ${weights15}
do
  params="+model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=finetune-convpool trainer.gpus=[0,1,2,3] train=False"
  python run.py experiment=cb55_select_train15_unet_finetune_rotnet     ${params}
done

for weight in ${weights30}
do
  params="+model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=finetune-convpool trainer.gpus=[0,1,2,3] train=False"
  python run.py experiment=cb55_select_train30_unet_finetune_rotnet     ${params}
done

