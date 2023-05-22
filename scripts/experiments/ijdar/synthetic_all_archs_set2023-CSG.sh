#!/usr/bin/env bash

set -e

#params_unet="experiment=synthetic_set2023-CSG_unet.yaml
#          trainer.devices=[0,1,2,3]
#          trainer.max_epochs=100
#          mode=ijdar.yaml
#          name=PT_sem_seg_synthetic_set2023-CSG_training-120_unet_loss_no_weights_100ep
#          logger.wandb.tags=[unet,set2023-CSG,training-120,4-classes,pre-training,100-epochs,no-weights,no-init]
#          logger.wandb.project=ijdar_controlled
#          logger.wandb.group=set2023-CSG-pt-100ep-unet"
#python run.py ${params_unet}
#
#params_unet="experiment=synthetic_set2023-CSG_adaptive_unet.yaml
#          trainer.devices=[0,1,2,3]
#          trainer.max_epochs=100
#          mode=ijdar.yaml
#          name=PT_sem_seg_synthetic_set2023-CSG_training-120_adaptive_unet_loss_no_weights_100ep
#          logger.wandb.tags=[adaptive_unet,set2023-CSG,training-120,4-classes,pre-training,100-epochs,no-weights,no-init]
#          logger.wandb.project=ijdar_controlled
#          logger.wandb.group=set2023-CSG-pt-100ep-adaptive_unet"
#python run.py ${params_unet}

params_unet="experiment=synthetic_set2023-CSG_doc_ufcn.yaml
          trainer.devices=[0,1,2,3]
          trainer.max_epochs=100
          mode=ijdar.yaml
          name=PT_sem_seg_synthetic_set2023-CSG_training-120_doc_ufcn_loss_no_weights_100ep-validation-RGB
          logger.wandb.tags=[doc_ufcn,set2022-CB,training-120,4-classes,pre-training,100-epochs,no-weights,no-init]
          logger.wandb.project=ijdar_controlled
          logger.wandb.group=set2023-CSG-pt-100ep-doc_ufcn"
python run.py ${params_unet}
