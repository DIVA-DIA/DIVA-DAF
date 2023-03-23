#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/23-21-13/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/23-32-16/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/23-43-08/checkpoints/epoch\=21/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-20/23-54-01/checkpoints/epoch\=30/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-21/00-04-53/checkpoints/epoch\=28/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-21/00-15-48/checkpoints/epoch\=14/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-21/00-26-39/checkpoints/epoch\=33/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-21/00-37-39/checkpoints/epoch\=11/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-21/00-48-36/checkpoints/epoch\=9/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/3cl_rlsa_vh_cb55_sauvola_unet_loss_no_weights_50epoch/2023-01-21/00-59-31/checkpoints/epoch\=24/backbone.pth")

for j in ${weights[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
          trainer.devices=[4,5,6,7]
          datamodule.train_folder_name=training-20
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_3cl_cb55_AB1_training-20_unet_loss_no_weights_50pt_100e
          logger.wandb.tags=[unet,AB1,training-20,3-classes,fine-tune,100-epochs,no-weights,3cl,rlsa_vh_3cl,50-epoch-pt]
          logger.wandb.group=fine-tune-sauvola-rlsa-vh-3cl-50pt-training-20"
  python run.py ${params}
done
