#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-06-09/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-22-35/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-39-00/checkpoints/epoch\=22/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/11-55-25/checkpoints/epoch\=16/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/12-12-00/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/12-28-38/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/12-45-12/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/13-01-53/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/13-18-31/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/binary_cb55_otsu_unet_loss_no_weights_50epochs/2023-01-21/13-35-08/checkpoints/epoch\=26/backbone.pth")

for j in ${weights[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[4,5,6,7]
          datamodule.train_folder_name=training-20
          +model.backbone.path_to_weights=${j}
          name=fine_tune_otsu_cb55_AB1_training-20_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,AB1,training-20,3-classes,fine-tune,100-epochs,no-weights,otsu,binary,50-epoch-pt]
          logger.wandb.group=fine-tune-otsu-50pt-training-20"
  python run.py ${params}
done
