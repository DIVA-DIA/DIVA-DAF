#!/usr/bin/env bash

set -e



for j in {0..4}; do
  params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_x5_setM_cb55_AB1_training-10_run_unet-100ep
        +model.backbone.path_to_weights=/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_setM_unet_loss_no_weights_200ep/2023-03-03/11-12-35/checkpoints/epoch\=74/backbone.pth
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[unet,AB1,training-10,fine-tune,setM-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=x5-setM-FT-training-10-100ep"
  #    echo ${params_unet}
  python run.py ${params_unet}
done

weights_backbone=("/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_setM_unet_loss_no_weights_200ep/2023-03-03/11-12-35/checkpoints/epoch\=74/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_setM_unet_loss_no_weights_200ep/2023-03-03/13-55-32/checkpoints/epoch\=133/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_setM_unet_loss_no_weights_200ep/2023-03-03/14-40-14/checkpoints/epoch\=117/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_setM_unet_loss_no_weights_200ep/2023-03-03/15-29-28/checkpoints/epoch\=93/backbone.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_setM_unet_loss_no_weights_200ep/2023-03-03/16-14-56/checkpoints/epoch\=60/backbone.pth"
)

for i in "${!weights_backbone[@]}"; do
  params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_setM_x5_cb55_AB1_training-10_run_unet-100ep-test
        +model.backbone.path_to_weights=${weights_backbone[i]}
        datamodule.train_folder_name=training-10
        logger.wandb.tags=[unet,AB1,training-10,fine-tune,x5-setM,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=setM-x5-FT-training-10-100ep"
  python run.py ${params_unet}
done