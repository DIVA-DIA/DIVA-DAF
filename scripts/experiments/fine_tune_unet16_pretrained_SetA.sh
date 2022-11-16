#!/usr/bin/env bash

set -e

weights_setA=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/sem_seg_synthetic_D60-setA_unet16_loss_no_weights/2022-11-10/12-34-49/checkpoints/epoch\=199/backbone.pth")
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet16_loss_no_weights/2022-11-01/13-51-50/checkpoints/epoch\=194/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet16_loss_no_weights/2022-11-01/16-33-58/checkpoints/epoch\=199/backbone.pth")

training_set=("training-20")
#  "training-20")
#  "training-40")

epochs=(100 100)

for j in "${!weights_setA[@]}"; do
  for i in "${!training_set[@]}"; do
    devices="[4,5]"
    if [ "${training_set[i]}" == "training-10" ]; then
      devices="[4]"
    fi
    params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet16.yaml
        trainer.devices=${devices}
        trainer.max_epochs=${epochs[i]}
        name=fine_tune_D60-SetA_cb55_AB1_${training_set[i]}_run_unet16
        model.backbone.path_to_weights=${weights_setA[j]}
        datamodule.train_folder_name=${training_set[i]}
        logger.wandb.tags=[unet16,AB1,${training_set[i]},fine-tune,D60-SetA,4-classes,baseline,${epochs[i]}-epochs,no-weights,best-jaccard]
        logger.wandb.group=D60-SetA-${training_set[i]}"
    python run.py ${params_unet}
  done
done
