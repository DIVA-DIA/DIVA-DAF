#!/usr/bin/env bash
set -e

weights="/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_pool/2021-12-16/17-28-04/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_pool/2021-12-16/18-07-05/checkpoints/backbone_last.pth
         /net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/rotnet_cb55_full_unet_pool/2021-12-16/18-46-26/checkpoints/backbone_last.pth"

seed="255827881"

gpus="[0,1,2,3]"

for weight in ${weights}
do
  params="+seed=${seed} +model.backbone.path_to_weights=\"${weight}\" logger.wandb.group=finetune-pool trainer.gpus=\"${gpus}\""
  python run.py experiment=cb55_select_train1_val1_run_unet_finetune_rotnet ${params}
  python run.py experiment=cb55_select_train15_run_unet_finetune_rotnet     ${params}
  python run.py experiment=cb55_full_run_unet_finetune_rotnet               ${params}
done

