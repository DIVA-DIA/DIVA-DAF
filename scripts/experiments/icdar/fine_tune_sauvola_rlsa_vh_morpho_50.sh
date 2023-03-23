#!/usr/bin/env bash

set -e

weights_rlsa_vh=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/21-21-50/checkpoints/epoch\=36/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/21-33-17/checkpoints/epoch\=23/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/21-45-43/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/21-57-21/checkpoints/epoch\=24/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/22-09-36/checkpoints/epoch\=44/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/22-21-07/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/22-32-44/checkpoints/epoch\=26/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/22-44-11/checkpoints/epoch\=31/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/22-55-35/checkpoints/epoch\=39/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/binary_cb55_sauvola_rlsa_vh_unet_loss_no_weights_50epochs/2023-01-18/23-07-00/checkpoints/epoch\=15/backbone.pth")

weights_morpho=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/19-07-10/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/19-18-34/checkpoints/epoch\=40/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/19-29-46/checkpoints/epoch\=45/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/19-41-32/checkpoints/epoch\=42/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/19-52-44/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/20-03-58/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/20-15-06/checkpoints/epoch\=34/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/20-27-03/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/20-38-15/checkpoints/epoch\=38/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/morpho_cb55_B22_unet_loss_no_weights_50epochs/2023-01-18/20-50-07/checkpoints/epoch\=39/backbone.pth")

for j in ${weights_rlsa_vh[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet_no_header_w.yaml
          trainer.devices=[4,5,6,7]
          datamodule.train_folder_name=training-20
          +model.backbone.path_to_weights=${j}
          name=fine_tune_sauvola_rlsa_vh_cb55_AB1_training-20_unet_loss_no_weights_50pt_100e_no_head
          logger.wandb.tags=[unet,AB1,training-20,3-classes,fine-tune,100-epochs,no-weights,sauvola_cleaned,binary_rlsa_vh,rlsa_vh,50-epoch-pt,no_head]
          logger.wandb.group=fine-tune-sauvola-rlsa-50pt-training-20"
  python run.py ${params}
done

for j in ${weights_morpho[*]}; do
  params="experiment=fine_tune_cb55_AB1_train_20_run_unet_no_header_w.yaml
          trainer.devices=[4,5,6,7]
          datamodule.train_folder_name=training-20
          +model.backbone.path_to_weights=${j}
          name=fine_tune_morpho_B22_cb55_AB1_training-20_unet_loss_no_weights_50pt_100e_no_head
          logger.wandb.tags=[unet,AB1,training-20,3-classes,fine-tune,100-epochs,no-weights,morpho,B22,50-epoch-pt,no_head]
          logger.wandb.group=fine-tune-morpho-50pt-training-20"
  python run.py ${params}
done
