#!/usr/bin/env bash

set -e

dataset_split=("training-10")
#  "training-20"
#  "training-40")

for split in "${dataset_split[@]}"; do
  for j in {1..3}; do
    params_unet="experiment=cb55_AB1_train_20_run_unet.yaml
        trainer.devices=[0,1]
        name=sem_seg_baseline_cb55_AB1_loss_no_weights_unet
        logger.wandb.tags=[unet,AB1,4-classes,baseline,300-epochs,no-weights,${split}]
        datamodule.train_folder_name=${split}
        trainer.max_epochs=300"

    python run.py ${params_unet}
  done
done
