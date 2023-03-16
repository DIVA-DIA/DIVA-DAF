#!/usr/bin/env bash

set -e

#weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_unet_loss_no_weights_100ep/2023-03-09/13-53-49/checkpoints/epoch\=68/backbone.pth"
#
#for j in {0..4}; do
#  params_unet="experiment=fine_tune_csg18_new_split_run_unet.yaml
#        trainer.devices=[0]
#        +trainer.precision=32
#        trainer.max_epochs=100
#        mode=ijdar.yaml
#        name=fine_tune_x5_set2023_csg_csg_new_split_training-10_run_unet-100ep
#        +model.backbone.path_to_weights=${weight}
#        logger.wandb.tags=[unet,AB1,training-10,fine-tune,set2023-csg,CSG-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
#        logger.wandb.project=ijdar_controlled
#        logger.wandb.group=x5-set2023-csg-FT-training-10-100ep"
#  #    echo ${params_unet}
#  python run.py ${params_unet}
#done
#
#weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_adaptive_unet_loss_no_weights_100ep/2023-03-09/14-35-01/checkpoints/epoch\=83/backbone.pth"
#
#for j in {0..4}; do
#  params_unet="experiment=fine_tune_csg18_new_split_run_adaptive_unet.yaml
#        trainer.devices=[0]
#        +trainer.precision=32
#        trainer.max_epochs=100
#        mode=ijdar.yaml
#        name=fine_tune_x5_set2023_csg_csg_new_split_training-10_run_adaptive_unet-100ep
#        +model.backbone.path_to_weights=${weight}
#        logger.wandb.tags=[adaptive_unet,AB1,training-10,fine-tune,set2023-csg,CSG-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
#        logger.wandb.project=ijdar_controlled
#        logger.wandb.group=x5-set2023-csg-FT-training-10-100ep"
#  #    echo ${params_unet}
#  python run.py ${params_unet}
#done

weight="/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_doc_ufcn_loss_no_weights_100ep-2/2023-03-14/07-36-32/checkpoints/epoch\=95/backbone.pth"

for j in {0..4}; do
  params_unet="experiment=fine_tune_csg18_new_split_run_doc_ufcn.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_x5_set2023_csg_csg_new_split_training-10_run_doc_ufcn-100ep-2
        +model.backbone.path_to_weights=${weight}
        logger.wandb.tags=[doc_ufcn,AB1,training-10,fine-tune,set2023-csg,CSG-x5,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=x5-set2023-csg-FT-training-10-100ep"
  #    echo ${params_unet}
  python run.py ${params_unet}
done
