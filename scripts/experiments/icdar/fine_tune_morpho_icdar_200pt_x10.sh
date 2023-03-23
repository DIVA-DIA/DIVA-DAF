#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/16-39-47/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/17-21-31/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/18-10-31/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/18-59-57/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/19-49-53/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/20-37-23/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/21-24-20/checkpoints/epoch\=48/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/22-13-27/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/23-00-32/checkpoints/epoch\=29/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_200epochs/2023-01-19/23-47-39/checkpoints/epoch\=31/backbone.pth")

for j in ${weights[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[0,1,2,3]
          datamodule.train_folder_name=training-20
          +model.backbone.path_to_weights=${j}
          name=fine_tune_morpho_B22_cb55_AB1_training-20_unet_loss_no_weights_200pt_100e
          logger.wandb.tags=[unet,AB1,training-20,3-classes,fine-tune,100-epochs,no-weights,morpho,B22,200-epoch-pt]
          logger.wandb.group=fine-tune-morpho-200pt-training-20"
  python run.py ${params}
done
