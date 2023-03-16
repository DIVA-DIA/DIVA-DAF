#!/usr/bin/env bash

set -e

params_unet="experiment=synthetic_set2023-CSG_unet16.yaml
          trainer.devices=[0,1,2,3]
          trainer.max_epochs=100
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_set2023-CSG_training-120_unet16_loss_no_weights_100ep
          logger.wandb.tags=[unet16,set2023-CSG,training-120,4-classes,pre-training,100-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=set2023-CSG-pt-100ep-unet16"
python run.py ${params_unet}
