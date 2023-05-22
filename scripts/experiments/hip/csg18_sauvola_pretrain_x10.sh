#!/usr/bin/env bash

set -e

epochs=("10" "20" "30" "40" "50")

for e in ${epochs[*]}; do
  for i in {0..9}; do
    params="experiment=binary_csg18_sauvola_dataset1_unet.yaml
        trainer.devices=[0,1,2,3]
        trainer.max_epochs=${e}
        mode=hip.yaml
        name=PT_csg18_sauvola_unet16_loss_no_weights_${e}epoch
        logger.wandb.project=hip
        logger.wandb.tags=[best_model,3cl_rlsa,sauvola,3cl,pre-training,unet16,dataset_rlsa_3cl_1,CSG18-ssl-set,${e}-epochs]
        logger.wandb.group=pre-training-rlsa-3cl-csg18-${e}-ep"
    python run.py ${params}
  done
done
