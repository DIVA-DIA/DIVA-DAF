#!/usr/bin/env bash
set -e

weights="/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_convpool/2021-12-16/13-24-34/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_convpool/2021-12-16/14-35-46/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_convpool/2021-12-16/15-47-32/checkpoints/backbone_last.pth"

seed="255827881"

gpus="[4,5,6,7]"

for weight in ${weights}
do
  params="+seed=${seed} +model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=finetune-convpool trainer.gpus=\"${gpus}\""
  python run.py experiment=cb55_select_train1_val1_run_unet_finetune_rotnet ${params}
  python run.py experiment=cb55_select_train15_run_unet_finetune_rotnet     ${params}
  python run.py experiment=cb55_full_run_unet_finetune_rotnet               ${params}
done