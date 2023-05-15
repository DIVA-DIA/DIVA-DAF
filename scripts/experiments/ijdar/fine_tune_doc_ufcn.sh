#!/usr/bin/env bash

set -e

weights=("/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_doc_ufcn_loss_no_weights_100ep-validation/2023-03-27/13-31-28/checkpoints/epoch\=61/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_doc_ufcn_loss_no_weights_100ep-validation/2023-03-27/13-58-10/checkpoints/epoch\=98/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_doc_ufcn_loss_no_weights_100ep-validation/2023-03-27/14-24-43/checkpoints/epoch\=43/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_doc_ufcn_loss_no_weights_100ep-validation/2023-03-27/14-51-06/checkpoints/epoch\=62/backbone.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_luca/ijdar/PT_sem_seg_synthetic_set2023-CSG_training-120_doc_ufcn_loss_no_weights_100ep-validation/2023-03-27/15-18-08/checkpoints/epoch\=52/backbone.pth"
)

for w in "${!weights[@]}"; do
  params_unet="experiment=fine_tune_csg18_new_split_run_doc_ufcn.yaml
        trainer.devices=[0]
        +trainer.precision=32
        trainer.max_epochs=100
        mode=ijdar.yaml
        name=fine_tune_csg18_run_doc_ufcn-100ep-validation
        +model.backbone.path_to_weights=${weights[w]}
        logger.wandb.tags=[doc_ufcn,AB1,training-10,fine-tune,4-classes,baseline,100-epochs,no-weights,best-jaccard]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=csg18-validation-FT-100ep"
#      echo ${params_unet}
  python run.py ${params_unet}
done
