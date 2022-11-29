#!/usr/bin/env bash

set -e

weights_setA=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/10-50-47/checkpoints/backbone_last.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/13-27-29/checkpoints/backbone_last.pth"
  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/16-09-00/checkpoints/backbone_last.pth")

weights_setB=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/11-42-54/checkpoints/backbone_last.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/14-21-39/checkpoints/backbone_last.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/17-03-43/checkpoints/backbone_last.pth")

weights_setC=("/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/12-34-43/checkpoints/backbone_last.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/15-14-59/checkpoints/backbone_last.pth"
"/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet_loss_no_weights/2022-11-01/17-56-55/checkpoints/backbone_last.pth")


training_set=("training-10"
  "training-20"
  "training-40")
dataset="PC60-SetC"
#  "D60-SetA"
#  "DPC60-SetB")

epochs=(200 100 50)

for j in "${!weights_setC[@]}"; do
  for i in "${!training_set[@]}"; do
    params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[0,1,2,3]
        trainer.max_epochs=${epochs[i]}
        name=fine_tune_${dataset}_cb55_AB1_${training_set[i]}_run_unet
        model.backbone.path_to_weights=${weights_setC[j]}
        datamodule.train_folder_name=${training_set[i]}
        logger.wandb.tags=[unet,AB1,${training_set[i]},fine-tune,${dataset},4-classes,baseline,${epochs[i]}-epochs,no-weights]
        logger.wandb.group=D60-FT-SetA-${training_set[i]}"
    python run.py ${params_unet}
  done
done
