#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/12-26-00/checkpoints/epoch\=41/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/12-38-10/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/12-50-08/checkpoints/epoch\=41/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/13-02-21/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/13-14-30/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/13-26-30/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/13-38-35/checkpoints/epoch\=46/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/13-50-42/checkpoints/epoch\=49/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/14-02-44/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_cleaned_unet_loss_no_weights_50epoch/2023-01-14/14-14-48/checkpoints/epoch\=45/backbone.pth")

for j in ${weights[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[4,5,6,7]
          datamodule.train_folder_name=training-10
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_cleaned_cb55_AB1_training-10_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,AB1,training-10,3-classes,fine-tune,100-epochs,no-weights,sauvola_cleaned,binary,50-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-cleaned-50pt-training-10"
  python run.py ${params}
done
