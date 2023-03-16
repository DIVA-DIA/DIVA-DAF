#!/usr/bin/env bash

set -e

weights_backbone=("/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet_loss_no_weights-200ep/2023-03-02/13-19-50/checkpoints/epoch\=35/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet_loss_no_weights-200ep/2023-03-02/12-29-25/checkpoints/epoch=138/backbone.pth"
)
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet_loss_no_weights-200ep/2023-03-01/14-00-59/checkpoints/epoch\=42/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet_loss_no_weights-200ep/2023-03-01/15-39-37/checkpoints/epoch\=40/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet_loss_no_weights-200ep/2023-03-01/17-19-28/checkpoints/epoch\=43/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/morpho_cb55_B22_unet_loss_no_weights-200ep/2023-03-01/18-59-04/checkpoints/epoch\=41/backbone.pth"
#)

for i in "${!weights_backbone[@]}"; do
  params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[0]
        +trainer.precision=16
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_morpho_cb55_AB1_training-10_run_unet-100ep-test
        +model.backbone.path_to_weights=${weights_backbone[i]}
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[unet,AB1,training-10,fine-tune,morpho,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=morpho-FT-training-10-100ep"
  python run.py ${params_unet}
done
