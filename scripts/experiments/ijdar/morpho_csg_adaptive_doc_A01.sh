#!/usr/bin/env bash

set -e


params_unet="experiment=morpho_CSG_A01_unet.yaml
          trainer.devices=[0,1,2,3]
          trainer.max_epochs=200
          mode=ijdar.yaml
          name=PT_sem_seg_morpho_csg_A01_unet_loss_no_weights_200ep
          logger.wandb.tags=[unet,morpho-CSG,4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=morpho-csg-pt-200ep-unet"
python run.py ${params_unet}

params_unet="experiment=morpho_CSG_A01_unet16.yaml
          trainer.devices=[0,1,2,3]
          trainer.max_epochs=200
          mode=ijdar.yaml
          name=PT_morpho_csg_A01_unet16_loss_no_weights_200ep
          logger.wandb.tags=[unet16,morpho-CSG,4-classes,pre-training,200-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=morpho-csg-pt-200ep-unet16"
python run.py ${params_unet}
