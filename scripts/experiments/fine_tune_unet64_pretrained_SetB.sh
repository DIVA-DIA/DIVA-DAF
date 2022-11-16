#!/usr/bin/env bash

set -e

weights_setB=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/sem_seg_synthetic_DPC60-setB_unet64_loss_no_weights/2022-11-10/09-49-09/checkpoints/epoch\=196/backbone.pth")
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet64_loss_no_weights/2022-11-01/15-03-17/checkpoints/backbone_last.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet64_loss_no_weights/2022-11-01/17-45-26/checkpoints/backbone_last.pth")

training_set=("training-20")
#  "training-20")
#  "training-40")

epochs=(100 100)

for j in "${!weights_setB[@]}"; do
  for i in "${!training_set[@]}"; do
    devices="[4,5]"
    if [ "${training_set[i]}" == "training-10" ]; then
      devices="[4]"
    fi
    params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet64.yaml
        trainer.devices=${devices}
        trainer.max_epochs=${epochs[i]}
        name=fine_tune_DPC60-SetB_cb55_AB1_${training_set[i]}_run_unet64
        model.backbone.path_to_weights=${weights_setB[j]}
        datamodule.train_folder_name=${training_set[i]}
        logger.wandb.tags=[unet64,AB1,${training_set[i]},fine-tune,DPC60-SetB,4-classes,baseline,${epochs[i]}-epochs,no-weights,best-jaccard]
        logger.wandb.group=D60-FT-SetA-${training_set[i]}"
    python run.py ${params_unet}
  done
done
