#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/15-37-47/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/15-54-07/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/16-10-23/checkpoints/epoch\=41/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/16-26-50/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/16-43-21/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/16-59-39/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/17-16-01/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/17-32-26/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/17-48-49/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/3cl_rlsa_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/18-05-16/checkpoints/epoch\=16/backbone.pth")

training=("training-10" "training-5")

for t in ${training[*]}; do
  devices="[4,5,6,7]"
  if [ "${t}" == "training-10" ]; then
    devices="[4,5]"
  fi
  if [ "${t}" == "training-5" ]; then
    devices="[4]"
  fi
  for j in ${weights[*]}; do
    params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=${devices}
          datamodule.train_folder_name=${t}
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_3cl_cb55_AB1_${t}_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,AB1,${t},3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_3cl,50-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-3cl-50pt-${t}"
    python run.py ${params}
  done
done
