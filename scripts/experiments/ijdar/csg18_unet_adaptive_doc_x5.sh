#!/usr/bin/env bash

set -e

params_unet="experiment=csg18_new_split_train_10_run_unet.yaml
        trainer.devices=[0,1]
        +trainer.precision=32
        mode=ijdar.yaml
        name=sem_seg_baseline_csg18_new_loss_no_weights_unet_800ep
        logger.wandb.tags=[unet,csg18_new,4-classes,baseline,800-epochs,no-weights,training-10,time-measure]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=baseline_csg18_new-unet
        datamodule.train_folder_name=train
        trainer.max_epochs=800"

for i in {0..4}; do
  python run.py ${params_unet}
done

params_unet="experiment=csg18_new_split_train_10_run_adaptive_unet.yaml
        trainer.devices=[0,1]
        +trainer.precision=32
        mode=ijdar.yaml
        name=sem_seg_baseline_csg18_new_loss_no_weights_adaptive_unet_800ep
        logger.wandb.tags=[adaptive_unet,csg18_new,4-classes,baseline,800-epochs,no-weights,training-10,time-measure]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=baseline_csg18_new-adaptive_unet
        datamodule.train_folder_name=train
        trainer.max_epochs=800"

for i in {0..4}; do
  python run.py ${params_unet}
done

params_unet="experiment=csg18_new_split_train_10_run_doc_ufcn.yaml
        trainer.devices=[0,1]
        +trainer.precision=32
        mode=ijdar.yaml
        name=sem_seg_baseline_csg18_new_loss_no_weights_doc_ufcn_800ep
        logger.wandb.tags=[doc_ufcn,csg18_new,4-classes,baseline,800-epochs,no-weights,training-10,time-measure]
        logger.wandb.project=ijdar_controlled
        logger.wandb.group=baseline_csg18_new-doc_ufcn
        datamodule.train_folder_name=train
        trainer.max_epochs=800"

for i in {0..4}; do
  python run.py ${params_unet}
done
