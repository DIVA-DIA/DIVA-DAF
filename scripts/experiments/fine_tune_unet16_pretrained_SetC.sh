#!/usr/bin/env bash

set -e

weights_setC=("/net/research-hisdoc/experiments_lars_paul/lars_luca/experiments/sem_seg_synthetic_PC60-setC_unet16_loss_no_weights/2022-11-10/13-50-00/checkpoints/epoch\=156/backbone.pth")
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet16_loss_no_weights/2022-11-01/15-38-47/checkpoints/epoch\=138/backbone.pth"
#  "/net/research-hisdoc/experiments_lars_paul/lars_lucy/experiments/sem_seg_synthetic_DPC_60_unet16_loss_no_weights/2022-11-01/18-20-08/checkpoints/epoch\=196/backbone.pth")

training_set=("training-20")
#  "training-20")
#  "training-40")

epochs=(100 100)

for j in "${!weights_setC[@]}"; do
  for i in "${!training_set[@]}"; do
    devices="[4,5]"
    if [ "${training_set[i]}" == "training-10" ]; then
      devices="[4]"
    fi
    params_unet="experiment=fine_tune_cb55_AB1_train_20_run_unet16.yaml
        trainer.devices=${devices}
        trainer.max_epochs=${epochs[i]}
        name=fine_tune_PC60-SetC_cb55_AB1_${training_set[i]}_run_unet16
        model.backbone.path_to_weights=${weights_setC[j]}
        datamodule.train_folder_name=${training_set[i]}
        logger.wandb.tags=[unet16,AB1,${training_set[i]},fine-tune,PC60-SetC,4-classes,baseline,${epochs[i]}-epochs,no-weights,best-jaccard]
        logger.wandb.group=PC60-SetC-${training_set[i]}"
    python run.py ${params_unet}
  done
done
