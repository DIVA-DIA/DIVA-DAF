#!/usr/bin/env bash
set -e

weights="/netscratch/experiments_lars_paul/lars/experiments/rotnet_cb55_full_resnet50/2021-12-07/18-33-43/checkpoints/epoch=39/backbone.pth
         /netscratch/experiments_lars_paul/lars/experiments/rotnet_cb55_full_resnet50/2021-12-07/22-33-54/checkpoints/epoch=43/backbone.pth
         /netscratch/experiments_lars_paul/lars/experiments/rotnet_cb55_full_resnet50/2021-12-08/02-33-35/checkpoints/epoch=19/backbone.pth"

seed="255827881"

for weight in ${weights}
do
  params="+seed=\"${seed}\" +model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=\"finetune\""
  python run.py experiment=cb55_full_run_resnet50               name="semantic_segmentation_cb55_full_resnet50_finetune_rotnet"        ${params}
  python run.py experiment=cb55_select_train15_run_resnet50     name="semantic_segmentation_cb55_train15_resnet50_finetune_rotnet"     ${params}
  python run.py experiment=cb55_select_train1_val1_run_resnet50 name="semantic_segmentation_cb55_train1_val1_resnet50_finetune_rotnet" ${params}
done
