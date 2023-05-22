#!/usr/bin/env bash

set -e

weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_unet16_loss_no_weights_100ep/2023-03-08/12-24-40/checkpoints/epoch\=92/backbone.pth"

for j in {0..4}; do
  params_unet="experiment=fine_tune_csg18_new_split_run_unet16.yaml
        trainer.devices=[2]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_x5_set2023_csg_csg_new_split_training-10_run_unet16-100ep
        +model.backbone.path_to_weights=${weight}
        logger.wandb.tags=[unet16,AB1,training-10,fine-tune,set2023-csg,CSG-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=x5-set2023-csg-FT-training-10-100ep"
  #    echo ${params_unet}
  python run.py ${params_unet}
done
