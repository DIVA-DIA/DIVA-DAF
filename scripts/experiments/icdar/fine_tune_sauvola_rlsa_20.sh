#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/11-59-03/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-04-34/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-10-33/checkpoints/epoch\=18/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-16-11/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-21-55/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-27-35/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-33-20/checkpoints/epoch\=19/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-38-55/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-44-25/checkpoints/epoch\=17/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_unet_loss_no_weights_20epochs/2023-01-16/12-50-20/checkpoints/epoch\=15/backbone.pth")

for j in ${weights[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[4]
          datamodule.train_folder_name=training-5
          +model.backbone.path_to_weights=${j}
          name=fine_tune_morpho_B22_cb55_AB1_training-20_unet_loss_no_weights_50pt_100e_no_head
          logger.wandb.tags=[unet,AB1,training-5,3-classes,fine-tune,100-epochs,no-weights,sauvola_cleaned,binary_rlsa,rlsa,20-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-20pt-training-5"
  python run.py ${params}
done
